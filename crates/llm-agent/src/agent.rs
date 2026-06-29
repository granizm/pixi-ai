//! The agent — drives the agentic loop and carries conversation history.
//!
//! [`LlmAgent`] owns a [`crate::memory::ConversationMemory`] so multi-turn
//! conversations relay context: "did we have X?" → answer → "what about Y?" →
//! "so what's the conclusion?" each see the prior turns. Within a single
//! [`LlmAgent::respond`] call it runs the agentic loop: ask the provider, and
//! while the model emits `tool_use`, execute the tool and feed the result back,
//! until the model produces a final answer.

use llm::{ContentBlock, LlmProvider, Message, Role, StopReason, TurnRequest};

use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::memory::ConversationMemory;
use crate::tools::ToolRegistry;
use crate::types::AgentOutcome;

/// A stateful conversational agent over an [`LlmProvider`].
pub struct LlmAgent {
    config: AgentConfig,
    provider: Box<dyn LlmProvider>,
    tools: ToolRegistry,
    memory: ConversationMemory,
}

impl LlmAgent {
    /// Build an agent from config + a constructed provider + a tool registry.
    pub fn new(config: AgentConfig, provider: Box<dyn LlmProvider>, tools: ToolRegistry) -> Self {
        Self {
            config,
            provider,
            tools,
            memory: ConversationMemory::new(),
        }
    }

    /// Handle one user utterance (typically a whisper transcript).
    ///
    /// Appends the user turn to history, runs the agentic loop (tool round-trips
    /// up to [`AgentConfig::max_iterations`]), appends the assistant turn, and
    /// returns the final answer. Because history is retained, the *next* call
    /// can resolve references like "what about that?".
    pub async fn respond(&mut self, user_text: String) -> Result<AgentOutcome, AgentError> {
        self.memory.push(Message::user_text(user_text));

        let tool_defs = self.tools.definitions();

        for iteration in 0..self.config.max_iterations {
            let request = TurnRequest {
                messages: self.memory.messages(),
                system: self.config.system.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.config.max_tokens,
            };

            let response = self.provider.turn(&request).await?;

            // Record the assistant turn (text + any tool_use blocks) so history
            // and the next turn stay consistent.
            self.memory.push(Message {
                role: Role::Assistant,
                content: response.blocks.clone(),
            });

            match response.stop_reason {
                StopReason::EndTurn | StopReason::MaxTokens => {
                    let text = extract_text(&response.blocks);
                    return Ok(AgentOutcome {
                        text,
                        iterations: iteration + 1,
                    });
                }
                StopReason::ToolUse => {
                    // Execute every requested tool and gather the results into a
                    // single user turn, then continue the loop.
                    let result_blocks = self.run_tool_calls(&response.blocks).await?;
                    self.memory.push(Message {
                        role: Role::User,
                        content: result_blocks,
                    });
                    // continue to the next iteration
                }
            }
        }

        Err(AgentError::NotConverged(self.config.max_iterations))
    }

    /// Execute each `tool_use` block in `blocks` and return the corresponding
    /// `tool_result` blocks (in request order).
    async fn run_tool_calls(
        &self,
        blocks: &[ContentBlock],
    ) -> Result<Vec<ContentBlock>, AgentError> {
        let mut results = Vec::new();
        for block in blocks {
            let ContentBlock::ToolUse { id, name, input } = block else {
                continue;
            };
            let result_block = match self.tools.get(name) {
                Some(tool) => match tool.call(input.clone()).await {
                    Ok(output) => ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: output,
                        is_error: false,
                    },
                    Err(e) => ContentBlock::ToolResult {
                        tool_use_id: id.clone(),
                        content: e.to_string(),
                        is_error: true,
                    },
                },
                None => ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: format!("unknown tool: {name}"),
                    is_error: true,
                },
            };
            results.push(result_block);
        }
        Ok(results)
    }

    /// Reset the conversation (start a new topic).
    pub fn reset(&mut self) {
        self.memory.clear();
    }

    /// Read-only access to the conversation length (turns retained).
    pub fn history_len(&self) -> usize {
        self.memory.len()
    }
}

/// Concatenate the text of all [`ContentBlock::Text`] blocks (the model's final
/// natural-language answer), skipping empties.
fn extract_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for block in blocks {
        if let ContentBlock::Text { text } = block {
            if text.is_empty() {
                continue;
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

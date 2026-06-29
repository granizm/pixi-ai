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
    pub fn new(
        config: AgentConfig,
        provider: Box<dyn LlmProvider>,
        tools: ToolRegistry,
    ) -> Self {
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
    ///
    /// Implemented in the next session; the loop skeleton below documents the
    /// intended control flow.
    pub async fn respond(&mut self, user_text: String) -> Result<AgentOutcome, AgentError> {
        self.memory.push(Message::user_text(user_text));

        // Intended agentic loop (filled in next session):
        //
        // for iteration in 0..self.config.max_iterations {
        //     let req = TurnRequest {
        //         messages: self.memory.messages(),
        //         system: self.config.system.clone(),
        //         tools: self.tools.definitions(),
        //         max_tokens: self.config.max_tokens,
        //     };
        //     let resp = self.provider.turn(&req).await?;
        //     self.memory.push(Message { role: Role::Assistant, content: resp.blocks.clone() });
        //     match resp.stop_reason {
        //         StopReason::EndTurn | StopReason::MaxTokens => {
        //             return Ok(final answer extracted from resp.blocks);
        //         }
        //         StopReason::ToolUse => {
        //             // dispatch each ContentBlock::ToolUse via self.tools,
        //             // push a user Message of ToolResult blocks, continue.
        //         }
        //     }
        // }
        // Err(AgentError::NotConverged(self.config.max_iterations))

        let _ = (&self.provider, &self.tools, &self.config);
        let _ = (
            StopReason::EndTurn,
            Role::Assistant,
            ContentBlock::Text { text: String::new() },
            TurnRequest {
                messages: Vec::new(),
                system: None,
                tools: Vec::new(),
                max_tokens: 0,
            },
        );
        unimplemented!("LlmAgent::respond — agentic loop implemented in the next session")
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

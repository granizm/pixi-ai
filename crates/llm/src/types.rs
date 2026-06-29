//! Provider-agnostic message and tool types.
//!
//! These are the common representation that every [`crate::provider::LlmProvider`]
//! implementation maps to / from. Anthropic, Gemini and Ollama each have their
//! own wire format (e.g. Anthropic `input_schema` vs Gemini `parameters`); the
//! per-provider modules translate between those shapes and the types here so the
//! agent layer never sees a provider-specific structure.

use serde::{Deserialize, Serialize};

/// Conversation role for a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single turn in the conversation history.
///
/// The API is stateless: the full history is sent on every [`crate::provider::LlmProvider::turn`]
/// call. A message's `content` is a list of [`ContentBlock`]s so a single
/// assistant turn can interleave text and `tool_use`, and a single user turn can
/// carry `tool_result`s back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Convenience constructor for a plain-text user message (the common entry
    /// point from a whisper transcript).
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
}

/// A piece of message content.
///
/// This is the union that makes agentic loops possible: the model emits
/// [`ContentBlock::ToolUse`], the agent executes the tool, and feeds the result
/// back as [`ContentBlock::ToolResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Natural-language text.
    Text { text: String },
    /// The model is requesting a tool call.
    ///
    /// `id` correlates a `ToolUse` with the matching [`ContentBlock::ToolResult`]
    /// (Anthropic's `tool_use_id`). For providers that don't supply an id
    /// (some local models), the provider synthesises one.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The caller's result for a prior [`ContentBlock::ToolUse`].
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// Declaration of a tool the model may call.
///
/// Maps to Anthropic's tool object (`name` / `description` / `input_schema`) and
/// Gemini's `function_declarations` entry (`name` / `description` / `parameters`).
/// `input_schema` is a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A parsed tool invocation extracted from a [`ContentBlock::ToolUse`], handed to
/// the agent layer for execution. Kept separate from the wire block so the agent
/// API doesn't depend on `ContentBlock`'s serde shape.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Why the model stopped generating this turn — drives the agentic loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its response. Loop terminates.
    EndTurn,
    /// The model wants to call one or more tools. Execute them and continue.
    ToolUse,
    /// Hit the `max_tokens` ceiling — output may be truncated.
    MaxTokens,
}

/// Capabilities a provider/model exposes, so the agent layer can decide whether
/// native tool calling is available or a prompt-based fallback is needed.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// The provider supports structured (native) tool calling over its API.
    /// When `false`, the agent layer may wrap it with a prompt-based emulation
    /// (reserved for a later phase).
    pub native_tool_calling: bool,
    /// The provider supports SSE / incremental streaming of the response.
    pub streaming: bool,
}

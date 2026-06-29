//! The provider abstraction.
//!
//! [`LlmProvider`] is the single trait every backend implements — cloud HTTP
//! APIs (Anthropic, Gemini), a local server (Ollama), and, in a later phase,
//! in-process inference (candle / llama-cpp-rs). The agent layer drives the
//! agentic loop by calling [`LlmProvider::turn`] repeatedly; it never depends on
//! which concrete provider is behind the trait object.

use async_trait::async_trait;

use crate::types::{Capabilities, ContentBlock, Message, StopReason, ToolDefinition};

/// One request for a single model turn (one step of the agentic loop).
#[derive(Debug, Clone)]
pub struct TurnRequest {
    /// Full conversation history (the API is stateless — send it every turn).
    pub messages: Vec<Message>,
    /// Optional system prompt.
    pub system: Option<String>,
    /// Tools the model may call this turn. Empty = plain completion.
    pub tools: Vec<ToolDefinition>,
    /// Hard ceiling on output tokens (Anthropic requires this; other providers
    /// map it to their own field).
    pub max_tokens: u32,
}

/// One model turn's result.
#[derive(Debug, Clone)]
pub struct TurnResponse {
    /// The assistant's content blocks — may interleave [`ContentBlock::Text`]
    /// and [`ContentBlock::ToolUse`].
    pub blocks: Vec<ContentBlock>,
    /// Why generation stopped — drives the agent loop's continue/terminate
    /// decision.
    pub stop_reason: StopReason,
}

/// A backend that can produce model turns.
///
/// `Send + Sync` so providers can live behind `Arc<dyn LlmProvider>` and be
/// shared across the agent and (eventually) the JNI/FFI boundary.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Run one model turn. The agent layer calls this in a loop, feeding tool
    /// results back via [`TurnRequest::messages`] until the response stops with
    /// [`StopReason::EndTurn`].
    async fn turn(&self, request: &TurnRequest) -> Result<TurnResponse, crate::error::LlmError>;

    /// What this provider/model supports — lets the agent decide between native
    /// tool calling and a prompt-based fallback.
    fn capabilities(&self) -> Capabilities;

    /// Human-readable provider id for logging / config (`"anthropic"`,
    /// `"gemini"`, `"ollama"`).
    fn name(&self) -> &'static str;
}

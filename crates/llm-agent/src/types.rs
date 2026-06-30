//! Agent-level result types.

/// The outcome of handling one user turn through the agentic loop.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The final natural-language answer to surface to the user (display / TTS).
    /// Empty when `error` is set.
    pub text: String,
    /// How many agentic iterations (tool round-trips) it took to get here.
    pub iterations: u32,
    /// Set when the turn failed (provider/network/tool error). When `Some`, the
    /// agent produced no answer and this is a human-readable reason to surface
    /// to the user instead of leaving the UI spinning. `None` on success.
    pub error: Option<String>,
}

impl AgentOutcome {
    /// A successful answer.
    pub fn answer(text: String, iterations: u32) -> Self {
        Self { text, iterations, error: None }
    }

    /// A failed turn carrying a human-readable reason (no answer text).
    pub fn failure(error: String) -> Self {
        Self { text: String::new(), iterations: 0, error: Some(error) }
    }
}

/// An event emitted during the agentic loop, for callers that want to observe
/// progress (e.g. show "looking that up…" in the UI while a tool runs).
///
/// Reserved for a streaming/observability API in a later session; defined now so
/// the public surface is stable.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// The model produced intermediate text.
    Text(String),
    /// The model is calling a tool (name).
    ToolCall(String),
    /// A tool returned its result.
    ToolResult { name: String, ok: bool },
    /// The loop finished with a final answer.
    Done(AgentOutcome),
}

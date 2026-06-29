//! Agent-level result types.

/// The outcome of handling one user turn through the agentic loop.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    /// The final natural-language answer to surface to the user (display / TTS).
    pub text: String,
    /// How many agentic iterations (tool round-trips) it took to get here.
    pub iterations: u32,
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

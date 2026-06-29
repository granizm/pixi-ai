//! Errors for the agent layer.

use thiserror::Error;

/// Errors surfaced while running the agentic loop.
#[derive(Debug, Error)]
pub enum AgentError {
    /// A provider call failed.
    #[error("provider error: {0}")]
    Provider(#[from] llm::LlmError),

    /// A tool returned an error or could not be found.
    #[error("tool error: {0}")]
    Tool(String),

    /// The loop exceeded [`crate::config::AgentConfig::max_iterations`] without
    /// the model reaching a final answer.
    #[error("agentic loop did not converge within {0} iterations")]
    NotConverged(u32),
}

//! Agent configuration.

use llm::ProviderConfig;

/// Configuration for an [`crate::agent::LlmAgent`].
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Which provider/model backs this agent.
    pub provider: ProviderConfig,
    /// System prompt establishing the agent's persona and tool-use guidance.
    pub system: Option<String>,
    /// Hard ceiling on agentic-loop iterations (tool round-trips) before giving
    /// up. Prevents runaway loops on ambiguous, multi-step queries.
    pub max_iterations: u32,
    /// Output-token ceiling per turn.
    pub max_tokens: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            // Filled in by the caller; this default is a placeholder so the
            // struct is constructible in tests/examples.
            provider: ProviderConfig {
                kind: llm::ProviderKind::Anthropic,
                model: String::new(),
                api_key: None,
                base_url: None,
                max_tokens: 4096,
            },
            system: None,
            max_iterations: 8,
            max_tokens: 4096,
        }
    }
}

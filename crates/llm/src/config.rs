//! Provider configuration.
//!
//! Follows the same pattern as the rest of pixi-ai: values come from the
//! environment first (production / secrets), with a YAML file as a dev-time
//! fallback (mirrors `speech-capture`'s config approach). API keys are never
//! hard-coded.

use serde::{Deserialize, Serialize};

/// Which backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Gemini,
    Ollama,
}

/// Configuration for constructing a provider.
///
/// `api_key` is read from the environment by convention:
/// - Anthropic: `ANTHROPIC_API_KEY`
/// - Gemini: `GEMINI_API_KEY`
/// - Ollama: none required (local server; `base_url` defaults to
///   `http://localhost:11434`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    /// Model id (e.g. `claude-opus-4-8`, `gemini-2.5-flash`, `gemma3`).
    pub model: String,
    /// API key. Optional for Ollama. Loaded from env in production.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Override the provider base URL (mainly for Ollama / self-hosted gateways).
    #[serde(default)]
    pub base_url: Option<String>,
    /// Default output-token ceiling for turns that don't specify one.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_tokens() -> u32 {
    4096
}

impl ProviderKind {
    /// The conventional environment variable holding this provider's API key,
    /// or `None` for providers that don't need one (Ollama).
    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            ProviderKind::Anthropic => Some("ANTHROPIC_API_KEY"),
            ProviderKind::Gemini => Some("GEMINI_API_KEY"),
            ProviderKind::Ollama => None,
        }
    }
}

impl ProviderConfig {
    /// Resolve the API key: an explicit [`ProviderConfig::api_key`] wins,
    /// otherwise fall back to the conventional environment variable for this
    /// provider kind ([`ProviderKind::api_key_env_var`]). Returns `None` for
    /// providers that don't require a key (Ollama).
    pub fn resolved_api_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key {
            return Some(key.clone());
        }
        self.kind
            .api_key_env_var()
            .and_then(|var| std::env::var(var).ok())
            .filter(|s| !s.is_empty())
    }

    /// Whether this provider requires an API key. Used to validate config before
    /// constructing a provider.
    pub fn requires_api_key(&self) -> bool {
        self.kind.api_key_env_var().is_some()
    }
}

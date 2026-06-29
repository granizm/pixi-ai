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

impl ProviderConfig {
    /// Resolve the API key: explicit value wins, else the conventional env var
    /// for this provider kind. Returns `None` for providers that don't need one.
    ///
    /// NOTE: actual env reading is implemented in a later session; this returns
    /// the explicit `api_key` only for now.
    pub fn resolved_api_key(&self) -> Option<String> {
        self.api_key.clone()
    }
}

//! Ollama provider (local server over HTTP) — the path to local gemma.
//!
//! Ollama hosts local models (gemma, llama, …) and exposes an HTTP API, so from
//! this crate's perspective it's "just another HTTP provider" alongside
//! Anthropic and Gemini — no in-process inference runtime, no GPU-backend
//! feature juggling. This is what lets the local-LLM requirement land at low
//! complexity in the first phase.
//!
//! Wire-format notes (for the implementation session):
//! - prefer the native `POST /api/chat` endpoint (tool calling + streaming).
//! - tool calling depends on the model: tool-capable gemma variants return
//!   structured `tool_calls`; others need the prompt-based fallback (handled in
//!   `llm-agent`, a later phase).
//! - default base URL `http://localhost:11434`; no API key required.

use async_trait::async_trait;

use crate::config::ProviderConfig;
use crate::error::LlmError;
use crate::provider::{LlmProvider, TurnRequest, TurnResponse};
use crate::types::Capabilities;

/// Default Ollama server endpoint.
pub const DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// Ollama provider. Holds the resolved config and a shared HTTP client.
pub struct OllamaProvider {
    #[allow(dead_code)] // wired up in the implementation session
    config: ProviderConfig,
}

impl OllamaProvider {
    /// Construct from config. `base_url` defaults to [`DEFAULT_BASE_URL`].
    pub fn new(config: ProviderConfig) -> Result<Self, LlmError> {
        Ok(Self { config })
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn turn(&self, _request: &TurnRequest) -> Result<TurnResponse, LlmError> {
        // Implemented in the next session: map common types → /api/chat body,
        // POST, map tool_calls (when present) back to ContentBlock::ToolUse.
        unimplemented!("OllamaProvider::turn — implemented in the next session")
    }

    fn capabilities(&self) -> Capabilities {
        // Native tool calling is model-dependent on Ollama; the agent layer
        // consults this together with the configured model. Conservative
        // default until the implementation session refines per-model detection.
        Capabilities {
            native_tool_calling: true,
            streaming: true,
            // Some local multimodal models (e.g. gemma multimodal via Ollama)
            // can take audio, but this provider is text-only for now.
            audio_input: false,
        }
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

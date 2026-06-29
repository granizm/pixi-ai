//! Google Gemini (Generative Language API) provider (raw HTTP over reqwest).
//!
//! Talks to `generateContent` directly. Function calling is supported natively,
//! but the wire shape differs from Anthropic in ways the provider must absorb so
//! the agent layer sees only the common types:
//! - tools live under `tools[].function_declarations[]` with a `parameters` key
//!   (vs Anthropic's `input_schema`).
//! - the response carries a `function_call` part (vs `stop_reason == "tool_use"`).
//! - streaming splits tool-call args across deltas — they must be reassembled.
//!
//! These differences are the reason provider mapping lives behind the trait
//! rather than leaking into `llm-agent`.

use async_trait::async_trait;

use crate::config::ProviderConfig;
use crate::error::LlmError;
use crate::provider::{LlmProvider, TurnRequest, TurnResponse};
use crate::types::Capabilities;

/// Gemini provider. Holds the resolved config and a shared HTTP client.
pub struct GeminiProvider {
    #[allow(dead_code)] // wired up in the implementation session
    config: ProviderConfig,
}

impl GeminiProvider {
    /// Construct from config. Validates that an API key is resolvable.
    pub fn new(config: ProviderConfig) -> Result<Self, LlmError> {
        Ok(Self { config })
    }
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn turn(&self, _request: &TurnRequest) -> Result<TurnResponse, LlmError> {
        // Implemented in the next session: map common types → generateContent
        // body (function_declarations / parameters), POST, map function_call
        // parts back to ContentBlock::ToolUse.
        unimplemented!("GeminiProvider::turn — implemented in the next session")
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_tool_calling: true,
            streaming: true,
            // Gemini does support audio input, but this provider is text-only
            // until the audio path is implemented. Advertise false so the agent
            // doesn't route audio here yet.
            audio_input: false,
        }
    }

    fn name(&self) -> &'static str {
        "gemini"
    }
}

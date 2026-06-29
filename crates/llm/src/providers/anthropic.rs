//! Anthropic Messages API provider (raw HTTP over reqwest).
//!
//! Rust has no official Anthropic SDK, so this talks to `POST /v1/messages`
//! directly with reqwest + serde. Tool use is supported natively: the response
//! `content` carries `tool_use` blocks, and tool results are sent back as
//! `tool_result` blocks in a user message. The agentic loop itself lives in the
//! `llm-agent` crate; this provider implements a single turn.
//!
//! Wire-format notes (for the implementation session):
//! - `max_tokens` is REQUIRED.
//! - tools use `input_schema` (JSON Schema).
//! - native tool calling: `stop_reason == "tool_use"`.
//! - auth header `x-api-key`, plus `anthropic-version: 2023-06-01`.

use async_trait::async_trait;

use crate::config::ProviderConfig;
use crate::error::LlmError;
use crate::provider::{LlmProvider, TurnRequest, TurnResponse};
use crate::types::Capabilities;

/// Anthropic provider. Holds the resolved config and a shared HTTP client.
pub struct AnthropicProvider {
    #[allow(dead_code)] // wired up in the implementation session
    config: ProviderConfig,
}

impl AnthropicProvider {
    /// Construct from config. Validates that an API key is resolvable.
    pub fn new(config: ProviderConfig) -> Result<Self, LlmError> {
        Ok(Self { config })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn turn(&self, _request: &TurnRequest) -> Result<TurnResponse, LlmError> {
        // Implemented in the next session: build the /v1/messages body from the
        // common types, POST it, map the response content + stop_reason back.
        unimplemented!("AnthropicProvider::turn — implemented in the next session")
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_tool_calling: true,
            streaming: true,
        }
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

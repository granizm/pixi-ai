//! Anthropic Messages API provider (raw HTTP over reqwest).
//!
//! Rust has no official Anthropic SDK, so this talks to `POST /v1/messages`
//! directly with reqwest + serde. Tool use is supported natively: the response
//! `content` carries `tool_use` blocks, and tool results are sent back as
//! `tool_result` blocks in a user message. The agentic loop itself lives in the
//! `llm-agent` crate; this provider implements a single turn.
//!
//! Wire-format notes:
//! - `max_tokens` is REQUIRED.
//! - tools use `input_schema` (JSON Schema).
//! - native tool calling: `stop_reason == "tool_use"`.
//! - auth header `x-api-key`, plus `anthropic-version: 2023-06-01`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::ProviderConfig;
use crate::error::LlmError;
use crate::provider::{LlmProvider, TurnRequest, TurnResponse};
use crate::types::{Capabilities, ContentBlock, Role, StopReason};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic provider. Holds the resolved config, API key, base URL, and a
/// shared HTTP client.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    model: String,
    default_max_tokens: u32,
    http: reqwest::Client,
}

impl AnthropicProvider {
    /// Construct from config. Fails if no API key can be resolved (explicit or
    /// `ANTHROPIC_API_KEY`).
    pub fn new(config: ProviderConfig) -> Result<Self, LlmError> {
        let api_key = config.resolved_api_key().ok_or_else(|| {
            LlmError::Config(
                "Anthropic requires an API key (set ANTHROPIC_API_KEY or ProviderConfig::api_key)"
                    .to_string(),
            )
        })?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        Ok(Self {
            api_key,
            base_url,
            model: config.model,
            default_max_tokens: config.max_tokens,
            http: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn turn(&self, request: &TurnRequest) -> Result<TurnResponse, LlmError> {
        // This provider is text-only (no audio_input). Per the ContentBlock::Audio
        // seam contract, reject audio rather than silently dropping it.
        if request.messages.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::Audio { .. }))
        }) {
            return Err(LlmError::Unsupported(
                "Anthropic provider does not accept audio input (use whisper/STT first)"
                    .to_string(),
            ));
        }

        let max_tokens = if request.max_tokens > 0 {
            request.max_tokens
        } else {
            self.default_max_tokens
        };

        let body = WireRequest {
            model: &self.model,
            max_tokens,
            system: request.system.as_deref(),
            messages: request.messages.iter().map(WireMessage::from).collect(),
            tools: request.tools.iter().map(WireTool::from).collect(),
        };

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Transport(e.to_string()))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::Transport(e.to_string()))?;

        if !status.is_success() {
            return Err(LlmError::Api(format!("HTTP {status}: {text}")));
        }

        let wire: WireResponse = serde_json::from_str(&text)?;
        Ok(wire.into_turn_response())
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_tool_calling: true,
            streaming: true,
            audio_input: false,
        }
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

// --- Wire types: serialize/deserialize the /v1/messages shape ---------------

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: Vec<WireContentBlock<'a>>,
}

impl<'a> From<&'a crate::types::Message> for WireMessage<'a> {
    fn from(m: &'a crate::types::Message) -> Self {
        WireMessage {
            role: match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            },
            content: m.content.iter().map(WireContentBlock::from).collect(),
        }
    }
}

/// Serialized content block. Mirrors [`ContentBlock`] but borrows for zero-copy
/// outbound serialization.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentBlock<'a> {
    Text {
        text: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a serde_json::Value,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: &'a str,
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl<'a> From<&'a ContentBlock> for WireContentBlock<'a> {
    fn from(b: &'a ContentBlock) -> Self {
        match b {
            ContentBlock::Text { text } => WireContentBlock::Text { text },
            ContentBlock::ToolUse { id, name, input } => {
                WireContentBlock::ToolUse { id, name, input }
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => WireContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: *is_error,
            },
            // Unreachable: turn() rejects audio before serialization (this
            // provider is text-only). Map to empty text as an infallible fallback.
            ContentBlock::Audio { .. } => WireContentBlock::Text { text: "" },
        }
    }
}

#[derive(Serialize)]
struct WireTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a serde_json::Value,
}

impl<'a> From<&'a crate::types::ToolDefinition> for WireTool<'a> {
    fn from(t: &'a crate::types::ToolDefinition) -> Self {
        WireTool {
            name: &t.name,
            description: &t.description,
            input_schema: &t.input_schema,
        }
    }
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<OwnedContentBlock>,
    stop_reason: Option<String>,
}

impl WireResponse {
    fn into_turn_response(self) -> TurnResponse {
        let stop_reason = match self.stop_reason.as_deref() {
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            // "end_turn", "stop_sequence", refusal, or absent → treat as done.
            _ => StopReason::EndTurn,
        };
        TurnResponse {
            blocks: self.content.into_iter().map(Into::into).collect(),
            stop_reason,
        }
    }
}

/// Owned inbound content block (response side). Only the variants Anthropic can
/// return in an assistant turn are modelled; unknown types are ignored.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OwnedContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Any block type we don't model (e.g. thinking) — dropped.
    #[serde(other)]
    Other,
}

impl From<OwnedContentBlock> for ContentBlock {
    fn from(b: OwnedContentBlock) -> Self {
        match b {
            OwnedContentBlock::Text { text } => ContentBlock::Text { text },
            OwnedContentBlock::ToolUse { id, name, input } => {
                ContentBlock::ToolUse { id, name, input }
            }
            // Map unknown blocks to empty text so the loop still has a valid
            // assistant turn; they carry no actionable content for us.
            OwnedContentBlock::Other => ContentBlock::Text {
                text: String::new(),
            },
        }
    }
}

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
            // Empty system (settings screen with a blank field) → omit entirely.
            system: request.system.as_deref().filter(|s| !s.is_empty()),
            // Messages whose content is empty after block filtering (e.g. an
            // assistant turn that was only thinking blocks) are rejected by
            // the API — drop them from the request.
            messages: request
                .messages
                .iter()
                .map(WireMessage::from)
                .filter(|m| !m.content.is_empty())
                .collect(),
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
            // filter_map: blocks that would serialize invalid (empty text,
            // audio on a text-only provider) are dropped defensively — the
            // API rejects empty text blocks with HTTP 400.
            content: m
                .content
                .iter()
                .filter_map(WireContentBlock::try_from_block)
                .collect(),
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

impl<'a> WireContentBlock<'a> {
    /// Serialize a domain block, or `None` if it must not go on the wire:
    /// empty text blocks are rejected by the API (HTTP 400), and audio has no
    /// representation on this text-only provider (turn() rejects it upfront;
    /// this is the infallible fallback).
    fn try_from_block(b: &'a ContentBlock) -> Option<Self> {
        match b {
            ContentBlock::Text { text } if text.is_empty() => None,
            ContentBlock::Text { text } => Some(WireContentBlock::Text { text }),
            ContentBlock::ToolUse { id, name, input } => {
                Some(WireContentBlock::ToolUse { id, name, input })
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Some(WireContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: *is_error,
            }),
            ContentBlock::Audio { .. } => None,
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
            blocks: self
                .content
                .into_iter()
                .filter_map(OwnedContentBlock::into_content_block)
                .collect(),
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

impl OwnedContentBlock {
    /// Convert to a domain block, or `None` for blocks that must not be kept
    /// in history: unknown types (e.g. `thinking`) and empty text.
    ///
    /// The previous version mapped unknown blocks to empty text — replaying
    /// that in the next turn's history triggers HTTP 400
    /// "messages: text content blocks must be non-empty" (observed with
    /// claude-sonnet-5, whose responses carry thinking blocks; 2026-07-04).
    fn into_content_block(self) -> Option<ContentBlock> {
        match self {
            OwnedContentBlock::Text { text } if text.is_empty() => None,
            OwnedContentBlock::Text { text } => Some(ContentBlock::Text { text }),
            OwnedContentBlock::ToolUse { id, name, input } => {
                Some(ContentBlock::ToolUse { id, name, input })
            }
            OwnedContentBlock::Other => None,
        }
    }
}

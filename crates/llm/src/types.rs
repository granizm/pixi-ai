//! Provider-agnostic message and tool types.
//!
//! These are the common representation that every [`crate::provider::LlmProvider`]
//! implementation maps to / from. Anthropic, Gemini and Ollama each have their
//! own wire format (e.g. Anthropic `input_schema` vs Gemini `parameters`); the
//! per-provider modules translate between those shapes and the types here so the
//! agent layer never sees a provider-specific structure.

use serde::{Deserialize, Serialize};

/// Conversation role for a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single turn in the conversation history.
///
/// The API is stateless: the full history is sent on every [`crate::provider::LlmProvider::turn`]
/// call. A message's `content` is a list of [`ContentBlock`]s so a single
/// assistant turn can interleave text and `tool_use`, and a single user turn can
/// carry `tool_result`s back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Convenience constructor for a plain-text user message (the common entry
    /// point from a whisper transcript).
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
}

/// A piece of message content.
///
/// This is the union that makes agentic loops possible: the model emits
/// [`ContentBlock::ToolUse`], the agent executes the tool, and feeds the result
/// back as [`ContentBlock::ToolResult`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Natural-language text.
    Text { text: String },
    /// The model is requesting a tool call.
    ///
    /// `id` correlates a `ToolUse` with the matching [`ContentBlock::ToolResult`]
    /// (Anthropic's `tool_use_id`). For providers that don't supply an id
    /// (some local models), the provider synthesises one.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The caller's result for a prior [`ContentBlock::ToolUse`].
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    /// Raw audio input for multimodal models that consume sound directly
    /// (e.g. gemma multimodal, Gemini audio input), bypassing whisper/STT.
    ///
    /// **Seam, not yet wired:** no provider sends or accepts this today — the
    /// HTTP providers are text-only this phase. It exists so the `sound → LLM`
    /// direct path can be added later without changing `ContentBlock`'s shape or
    /// breaking the text path. A provider that can't handle audio
    /// ([`Capabilities::audio_input`] = `false`) must reject a request
    /// containing this block rather than silently drop it.
    Audio {
        /// Encoded audio bytes (e.g. WAV/FLAC) or raw PCM, per `format`.
        #[serde(with = "base64_bytes")]
        data: Vec<u8>,
        /// MIME-ish format hint, e.g. `"audio/wav"`, `"audio/pcm;rate=16000"`.
        format: String,
    },
}

/// Serde helper: encode `Vec<u8>` audio as base64 so it round-trips through JSON
/// wire formats. (Used only by [`ContentBlock::Audio`], which is not yet sent.)
mod base64_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        // Minimal inline base64 (std has no base64); fine for the seam since it
        // isn't exercised yet. Replaced with a real encoder when audio lands.
        s.serialize_str(&encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        decode(&s).map_err(serde::de::Error::custom)
    }

    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn encode(input: &[u8]) -> String {
        let mut out = String::new();
        for chunk in input.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
            out.push(ALPHABET[(n >> 18 & 63) as usize] as char);
            out.push(ALPHABET[(n >> 12 & 63) as usize] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(n >> 6 & 63) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }

    fn decode(input: &str) -> Result<Vec<u8>, &'static str> {
        let val = |c: u8| -> Result<u32, &'static str> {
            ALPHABET
                .iter()
                .position(|&a| a == c)
                .map(|p| p as u32)
                .ok_or("invalid base64 char")
        };
        let cleaned: Vec<u8> = input.bytes().filter(|&c| c != b'=').collect();
        let mut out = Vec::new();
        for chunk in cleaned.chunks(4) {
            let mut n = 0u32;
            for (i, &c) in chunk.iter().enumerate() {
                n |= val(c)? << (18 - 6 * i);
            }
            out.push((n >> 16 & 0xff) as u8);
            if chunk.len() > 2 {
                out.push((n >> 8 & 0xff) as u8);
            }
            if chunk.len() > 3 {
                out.push((n & 0xff) as u8);
            }
        }
        Ok(out)
    }
}

/// Declaration of a tool the model may call.
///
/// Maps to Anthropic's tool object (`name` / `description` / `input_schema`) and
/// Gemini's `function_declarations` entry (`name` / `description` / `parameters`).
/// `input_schema` is a JSON Schema object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A parsed tool invocation extracted from a [`ContentBlock::ToolUse`], handed to
/// the agent layer for execution. Kept separate from the wire block so the agent
/// API doesn't depend on `ContentBlock`'s serde shape.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Why the model stopped generating this turn — drives the agentic loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its response. Loop terminates.
    EndTurn,
    /// The model wants to call one or more tools. Execute them and continue.
    ToolUse,
    /// Hit the `max_tokens` ceiling — output may be truncated.
    MaxTokens,
}

/// Capabilities a provider/model exposes, so the agent layer can decide whether
/// native tool calling is available or a prompt-based fallback is needed.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// The provider supports structured (native) tool calling over its API.
    /// When `false`, the agent layer may wrap it with a prompt-based emulation
    /// (reserved for a later phase).
    pub native_tool_calling: bool,
    /// The provider supports SSE / incremental streaming of the response.
    pub streaming: bool,
    /// The provider/model can consume [`ContentBlock::Audio`] directly
    /// (multimodal `sound → LLM`, no whisper/STT step). `false` for every
    /// provider this phase — it's the flag the bridge will consult later to
    /// decide whether to run STT or pass raw audio through.
    pub audio_input: bool,
}

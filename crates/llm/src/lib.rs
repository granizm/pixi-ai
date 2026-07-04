//! LLM provider abstraction for pixi-ai.
//!
//! This crate is the **base layer**: a single [`provider::LlmProvider`] trait
//! that every backend implements, plus the provider-agnostic [`types`] used to
//! talk to it. The **agentic loop** (tool execution, "did we have X? what about
//! Y? so what's the conclusion") lives in the sibling `llm-agent` crate, which
//! depends on this one.
//!
//! ## Providers
//!
//! All first-phase providers are HTTP clients (cloud or local server), gated by
//! Cargo features so a consumer pulls only what it needs:
//! - `anthropic` — Anthropic Messages API
//! - `gemini` — Google Generative Language API
//! - `ollama` — local Ollama server (the path to local gemma)
//!
//! In-process inference (candle / llama-cpp-rs) is intentionally deferred to a
//! later phase; the trait is shaped so it slots in without breaking callers.
//!
//! ## Cross-platform
//!
//! Like `speech-capture`, this crate is plain Rust and is consumed by the
//! Android (JNI) and iOS/desktop layers via path dependency, so a single
//! implementation serves all platforms.

pub mod config;
pub mod error;
pub mod provider;
pub mod providers;
pub mod types;

pub use config::{ProviderConfig, ProviderKind};
pub use error::LlmError;
pub use provider::{LlmProvider, TurnRequest, TurnResponse};
pub use types::{Capabilities, ContentBlock, Message, Role, StopReason, ToolCall, ToolDefinition};

/// Construct a boxed provider from configuration.
///
/// Dispatches on [`ProviderConfig::kind`] to the feature-gated implementation.
/// If the corresponding feature isn't enabled, returns [`LlmError::Config`].
pub fn build_provider(config: ProviderConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    match config.kind {
        #[cfg(feature = "anthropic")]
        ProviderKind::Anthropic => Ok(Box::new(providers::anthropic::AnthropicProvider::new(
            config,
        )?)),
        #[cfg(feature = "gemini")]
        ProviderKind::Gemini => Ok(Box::new(providers::gemini::GeminiProvider::new(config)?)),
        #[cfg(feature = "ollama")]
        ProviderKind::Ollama => Ok(Box::new(providers::ollama::OllamaProvider::new(config)?)),
        #[allow(unreachable_patterns)]
        other => Err(LlmError::Config(format!(
            "provider {:?} requested but its Cargo feature is not enabled",
            other
        ))),
    }
}

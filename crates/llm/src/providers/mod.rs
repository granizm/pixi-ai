//! Concrete provider implementations, each gated behind its own feature.
//!
//! Every module here maps the provider's wire format to/from the common
//! [`crate::types`] representation and implements [`crate::provider::LlmProvider`].

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;

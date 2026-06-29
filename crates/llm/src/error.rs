//! Error type shared across providers.

use thiserror::Error;

/// Errors surfaced by an [`crate::provider::LlmProvider`].
#[derive(Debug, Error)]
pub enum LlmError {
    /// Network / transport failure talking to an HTTP provider.
    #[error("http transport error: {0}")]
    Transport(String),

    /// The provider returned a non-success status or an error body.
    #[error("provider returned an error: {0}")]
    Api(String),

    /// Failed to (de)serialize a request or response.
    #[error("serialization error: {0}")]
    Serde(String),

    /// Required configuration (API key, base URL, model) was missing.
    #[error("configuration error: {0}")]
    Config(String),

    /// The requested capability (e.g. native tool calling) is not supported by
    /// this provider/model.
    #[error("unsupported capability: {0}")]
    Unsupported(String),
}

impl From<serde_json::Error> for LlmError {
    fn from(e: serde_json::Error) -> Self {
        LlmError::Serde(e.to_string())
    }
}

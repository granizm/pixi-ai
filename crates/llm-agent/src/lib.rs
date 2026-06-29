//! Agentic loop over an LLM provider.
//!
//! This is the **upper layer** on top of the `llm` crate. It turns a whisper
//! transcript into an answer by driving the model through tool-use turns —
//! handling conversational, multi-step queries like "did we have X? what about
//! Y? so what's the conclusion?".
//!
//! ## What lives here
//!
//! - [`agent::LlmAgent`] — a stateful conversational agent. Owns the provider, a
//!   [`tools::ToolRegistry`], and conversation history, and runs the agentic
//!   loop in [`agent::LlmAgent::respond`].
//! - [`tools::Tool`] — the interface for the capabilities the model can call to
//!   investigate ("did we have X?").
//! - [`memory::ConversationMemory`] — multi-turn history with a seam for future
//!   compaction (summarise + evict old turns).
//!
//! ## Relationship to `llm`
//!
//! Provider selection happens via this crate's features, which forward to the
//! base `llm` crate (`anthropic` / `gemini` / `ollama`). The agent talks only to
//! the [`llm::LlmProvider`] trait, so swapping providers — or adding in-process
//! inference later — doesn't touch the loop.
//!
//! ## Integration (next session)
//!
//! The intended entry point from the pipeline is the whisper transcript:
//! `speech-capture` emits a final transcript on `SpeechEvent::VoiceEnd`, which is
//! fed to [`agent::LlmAgent::respond`]; the resulting answer is surfaced to the
//! UI / TTS. On Android this is reached over JNI (a `llmPollResponse`-style poll
//! alongside the existing `sttPollTranscript`); on desktop via the app's update
//! loop. Wiring is deferred to the implementation session.

pub mod agent;
pub mod config;
pub mod error;
pub mod memory;
pub mod tools;
pub mod types;

/// Transcript bridge (channel + sequential async runner). The core
/// ([`speech::bridge`] / [`speech::SpeechRunner`]) needs only the `bridge`
/// feature; the `speech` feature adds [`speech::speech_callback`] for
/// speech-capture.
#[cfg(feature = "bridge")]
pub mod speech;

pub use agent::LlmAgent;
pub use config::AgentConfig;
pub use error::AgentError;
pub use memory::ConversationMemory;
pub use tools::{Tool, ToolError, ToolRegistry};
pub use types::{AgentEvent, AgentOutcome};

// Re-export the base layer so consumers can build providers and types without a
// separate dependency on `llm`.
pub use llm;

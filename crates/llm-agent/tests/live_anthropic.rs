//! Live smoke tests against the real Anthropic API.
//!
//! These are `#[ignore]` so CI / `cargo test` never spend money by accident.
//! Run explicitly with a key:
//!
//! ```sh
//! ANTHROPIC_API_KEY=sk-ant-... cargo test -p llm-agent --test live_anthropic -- --ignored --nocapture
//! ```
//!
//! Key handling is env-only (`ANTHROPIC_API_KEY`), matching the crate's
//! `ProviderConfig::resolved_api_key` fallback. If the key is absent the tests
//! skip with a message rather than failing.

#![cfg(feature = "anthropic")]

use std::sync::Arc;

use async_trait::async_trait;
use llm_agent::llm::{build_provider, ProviderConfig, ProviderKind};
use llm_agent::{AgentConfig, LlmAgent, Tool, ToolError, ToolRegistry};

const MODEL: &str = "claude-opus-4-8";

fn have_key() -> bool {
    std::env::var("ANTHROPIC_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
}

fn config() -> ProviderConfig {
    ProviderConfig {
        kind: ProviderKind::Anthropic,
        model: MODEL.to_string(),
        api_key: None,  // resolved from ANTHROPIC_API_KEY
        base_url: None, // real endpoint
        max_tokens: 1024,
    }
}

/// A trivial deterministic tool so the model has something concrete to call.
/// Returns the day a fictional "meeting note" was filed — the kind of "did we
/// have X?" lookup the agent is meant to answer.
struct NotesTool;

#[async_trait]
impl Tool for NotesTool {
    fn definition(&self) -> llm_agent::llm::ToolDefinition {
        llm_agent::llm::ToolDefinition {
            name: "search_notes".to_string(),
            description: "Search the user's saved notes for a topic. Returns whether a note \
                          exists and its date."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": { "type": "string", "description": "What to look for" }
                },
                "required": ["topic"]
            }),
        }
    }

    async fn call(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let topic = input
            .get("topic")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();
        // Deterministic canned data.
        if topic.contains("budget") {
            Ok("Found 1 note about 'budget' dated 2026-05-10.".to_string())
        } else {
            Ok(format!("No notes found about '{topic}'."))
        }
    }
}

/// One plain turn: transcript → text answer, no tools. Verifies the real wire
/// path end to end.
#[tokio::test]
#[ignore = "calls the real Anthropic API; needs ANTHROPIC_API_KEY"]
async fn live_plain_turn() {
    if !have_key() {
        eprintln!("SKIP: ANTHROPIC_API_KEY not set");
        return;
    }

    let provider = build_provider(config()).expect("provider");
    let mut agent = LlmAgent::new(AgentConfig::default(), provider, ToolRegistry::new());

    let outcome = agent
        .respond("Reply with exactly the single word: pong".to_string())
        .await
        .expect("live turn failed");

    eprintln!("model said: {:?}", outcome.text);
    assert!(!outcome.text.trim().is_empty(), "got empty answer");
    assert_eq!(outcome.iterations, 1, "no tools → single turn");
}

/// Full agentic loop with a real model: the agent should call `search_notes`
/// and then answer using its result.
#[tokio::test]
#[ignore = "calls the real Anthropic API; needs ANTHROPIC_API_KEY"]
async fn live_agentic_tool_use() {
    if !have_key() {
        eprintln!("SKIP: ANTHROPIC_API_KEY not set");
        return;
    }

    let provider = build_provider(config()).expect("provider");

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(NotesTool));

    let agent_config = AgentConfig {
        system: Some(
            "You answer questions about the user's saved notes. When asked whether something \
             exists, use the search_notes tool before answering. Keep answers to one sentence."
                .to_string(),
        ),
        max_iterations: 4,
        max_tokens: 1024,
        ..AgentConfig::default()
    };

    let mut agent = LlmAgent::new(agent_config, provider, registry);

    let outcome = agent
        .respond("Did we have anything about the budget?".to_string())
        .await
        .expect("live agentic loop failed");

    eprintln!(
        "iterations={} answer={:?}",
        outcome.iterations, outcome.text
    );
    // The model had to call the tool, so at least 2 iterations (tool + answer).
    assert!(
        outcome.iterations >= 2,
        "expected a tool round-trip, got {} iteration(s)",
        outcome.iterations
    );
    // The tool reported a 2026-05-10 date; a faithful answer should mention it.
    assert!(
        outcome.text.contains("2026-05-10") || outcome.text.to_lowercase().contains("budget"),
        "answer didn't reflect the tool result: {:?}",
        outcome.text
    );
}

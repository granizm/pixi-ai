//! Agentic-loop control-flow tests using a scripted mock provider.
//!
//! These exercise `LlmAgent::respond` without any network: a `ScriptedProvider`
//! returns a pre-set sequence of turns, and a `RecordingTool` lets us assert the
//! loop actually executed tools and fed results back.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use llm_agent::llm::{
    Capabilities, ContentBlock, LlmProvider, StopReason, TurnRequest, TurnResponse,
};
use llm_agent::{AgentConfig, LlmAgent, Tool, ToolError, ToolRegistry};

/// A provider that returns a fixed script of turns, one per `turn()` call, and
/// records every request it received (so we can assert on the relayed history).
struct ScriptedProvider {
    script: Mutex<std::collections::VecDeque<TurnResponse>>,
    seen_requests: Mutex<Vec<TurnRequest>>,
    calls: AtomicUsize,
}

impl ScriptedProvider {
    fn new(script: Vec<TurnResponse>) -> Self {
        Self {
            script: Mutex::new(script.into()),
            seen_requests: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn turn(&self, request: &TurnRequest) -> Result<TurnResponse, llm_agent::llm::LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen_requests.lock().unwrap().push(request.clone());
        let next = self.script.lock().unwrap().pop_front();
        // When the script runs dry, keep emitting end_turn so a misbehaving loop
        // terminates instead of hanging the test.
        Ok(next.unwrap_or(TurnResponse {
            blocks: vec![ContentBlock::Text {
                text: "(script exhausted)".to_string(),
            }],
            stop_reason: StopReason::EndTurn,
        }))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_tool_calling: true,
            streaming: false,
            audio_input: false,
        }
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}

/// A tool that records its inputs and returns a canned answer.
struct RecordingTool {
    calls: Arc<Mutex<Vec<serde_json::Value>>>,
}

#[async_trait]
impl Tool for RecordingTool {
    fn definition(&self) -> llm_agent::llm::ToolDefinition {
        llm_agent::llm::ToolDefinition {
            name: "lookup".to_string(),
            description: "Look something up".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"]
            }),
        }
    }

    async fn call(&self, input: serde_json::Value) -> Result<String, ToolError> {
        self.calls.lock().unwrap().push(input);
        Ok("found it: yes".to_string())
    }
}

fn tool_use(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
    ContentBlock::ToolUse {
        id: id.to_string(),
        name: name.to_string(),
        input,
    }
}

fn text(t: &str) -> ContentBlock {
    ContentBlock::Text {
        text: t.to_string(),
    }
}

#[tokio::test]
async fn loop_executes_tool_then_returns_final_answer() {
    // Turn 1: model asks to call `lookup`. Turn 2: model gives the final answer.
    let provider = ScriptedProvider::new(vec![
        TurnResponse {
            blocks: vec![tool_use("tu_1", "lookup", serde_json::json!({"q": "X"}))],
            stop_reason: StopReason::ToolUse,
        },
        TurnResponse {
            blocks: vec![text("Yes, we had X.")],
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let tool_calls = Arc::new(Mutex::new(Vec::new()));
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(RecordingTool {
        calls: tool_calls.clone(),
    }));

    let mut agent = LlmAgent::new(AgentConfig::default(), Box::new(provider), registry);

    let outcome = agent.respond("did we have X?".to_string()).await.unwrap();

    assert_eq!(outcome.text, "Yes, we had X.");
    assert_eq!(outcome.iterations, 2, "one tool round-trip + final answer");
    // The tool was actually executed with the model's input.
    let recorded = tool_calls.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0], serde_json::json!({"q": "X"}));
}

#[tokio::test]
async fn conversation_relays_history_across_turns() {
    // First user turn resolves immediately; second turn must SEE the first.
    let provider = ScriptedProvider::new(vec![
        TurnResponse {
            blocks: vec![text("We had X.")],
            stop_reason: StopReason::EndTurn,
        },
        TurnResponse {
            blocks: vec![text("And Y too.")],
            stop_reason: StopReason::EndTurn,
        },
    ]);
    let provider = Arc::new(provider);

    // We need a handle to inspect seen_requests after the run.
    struct Proxy(Arc<ScriptedProvider>);
    #[async_trait]
    impl LlmProvider for Proxy {
        async fn turn(
            &self,
            request: &TurnRequest,
        ) -> Result<TurnResponse, llm_agent::llm::LlmError> {
            self.0.turn(request).await
        }
        fn capabilities(&self) -> Capabilities {
            self.0.capabilities()
        }
        fn name(&self) -> &'static str {
            self.0.name()
        }
    }

    let mut agent = LlmAgent::new(
        AgentConfig::default(),
        Box::new(Proxy(provider.clone())),
        ToolRegistry::new(),
    );

    agent.respond("did we have X?".to_string()).await.unwrap();
    agent.respond("what about Y?".to_string()).await.unwrap();

    let seen = provider.seen_requests.lock().unwrap();
    // The second provider call must carry the prior turns:
    // user(X), assistant(We had X.), user(Y) = 3 messages.
    let second = &seen[1];
    assert_eq!(
        second.messages.len(),
        3,
        "second turn relays the full prior conversation"
    );
    // history retained on the agent: u,a,u,a = 4
    assert_eq!(agent.history_len(), 4);
}

#[tokio::test]
async fn loop_gives_up_after_max_iterations() {
    // Provider always asks for a tool — the loop must cut off, never hang.
    let script: Vec<TurnResponse> = (0..10)
        .map(|i| TurnResponse {
            blocks: vec![tool_use(
                &format!("tu_{i}"),
                "lookup",
                serde_json::json!({ "q": i }),
            )],
            stop_reason: StopReason::ToolUse,
        })
        .collect();
    let provider = ScriptedProvider::new(script);

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(RecordingTool {
        calls: Arc::new(Mutex::new(Vec::new())),
    }));

    let config = AgentConfig {
        max_iterations: 3,
        ..AgentConfig::default()
    };

    let mut agent = LlmAgent::new(config, Box::new(provider), registry);
    let err = agent
        .respond("loop forever?".to_string())
        .await
        .unwrap_err();

    assert!(
        matches!(err, llm_agent::AgentError::NotConverged(3)),
        "expected NotConverged(3), got {err:?}"
    );
}

#[tokio::test]
async fn unknown_tool_is_reported_as_error_result() {
    // Model calls a tool that isn't registered; loop should feed back an error
    // tool_result (not panic), then the model concludes.
    let provider = ScriptedProvider::new(vec![
        TurnResponse {
            blocks: vec![tool_use("tu_1", "nonexistent", serde_json::json!({}))],
            stop_reason: StopReason::ToolUse,
        },
        TurnResponse {
            blocks: vec![text("Sorry, I couldn't do that.")],
            stop_reason: StopReason::EndTurn,
        },
    ]);

    let mut agent = LlmAgent::new(
        AgentConfig::default(),
        Box::new(provider),
        ToolRegistry::new(), // empty registry
    );

    let outcome = agent
        .respond("use a missing tool".to_string())
        .await
        .unwrap();
    assert_eq!(outcome.text, "Sorry, I couldn't do that.");
}

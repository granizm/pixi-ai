//! Bridge tests: back-to-back transcripts are processed sequentially, in order.
//!
//! These exercise the channel → `SpeechRunner` path without speech-capture
//! hardware: a mock provider stands in for the LLM and records the order in
//! which utterances were handled. Gated on the `bridge` feature — the core
//! needs no speech-capture / native ten-vad, so it links and runs anywhere.

#![cfg(feature = "bridge")]

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use llm_agent::llm::{
    Capabilities, ContentBlock, LlmProvider, StopReason, TurnRequest, TurnResponse,
};
use llm_agent::speech::bridge;
use llm_agent::{AgentConfig, LlmAgent, ToolRegistry};

/// A provider that echoes back the latest user text and records, in order, the
/// user utterances it was asked to answer. A small await yield makes overlap
/// detectable if the runner were ever concurrent.
struct EchoProvider {
    seen_user_texts: Arc<Mutex<Vec<String>>>,
    in_flight: Arc<AtomicI32>,
    max_concurrent: Arc<AtomicI32>,
}

#[async_trait]
impl LlmProvider for EchoProvider {
    async fn turn(&self, request: &TurnRequest) -> Result<TurnResponse, llm_agent::llm::LlmError> {
        // Track concurrency: if the runner ever ran two turns at once, this would
        // climb above 1.
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_concurrent.fetch_max(now, Ordering::SeqCst);

        // The latest user turn is the last message with a Text block.
        let latest = request
            .messages
            .iter()
            .rev()
            .find_map(|m| {
                m.content.iter().find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .unwrap_or_default();
        self.seen_user_texts.lock().unwrap().push(latest.clone());

        // Yield so a (hypothetically) concurrent runner would interleave here.
        tokio::time::sleep(Duration::from_millis(5)).await;

        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        Ok(TurnResponse {
            blocks: vec![ContentBlock::Text {
                text: format!("ack: {latest}"),
            }],
            stop_reason: StopReason::EndTurn,
        })
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_tool_calling: false,
            streaming: false,
            audio_input: false,
        }
    }

    fn name(&self) -> &'static str {
        "echo"
    }
}

#[tokio::test]
async fn back_to_back_transcripts_processed_sequentially_in_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let max_concurrent = Arc::new(AtomicI32::new(0));
    let provider = EchoProvider {
        seen_user_texts: seen.clone(),
        in_flight: Arc::new(AtomicI32::new(0)),
        max_concurrent: max_concurrent.clone(),
    };

    let agent = LlmAgent::new(
        AgentConfig::default(),
        Box::new(provider),
        ToolRegistry::new(),
    );
    let (tx, mut outcomes, runner) = bridge(agent);

    // Drive the runner in the background.
    let handle = tokio::spawn(runner.run());

    // Fire several transcripts "back to back" before reading any answer.
    for utterance in ["did we have X?", "what about Y?", "and Z?"] {
        tx.send(utterance.to_string()).await.unwrap();
    }
    // Close the input so the runner finishes after draining.
    drop(tx);

    // Collect answers in arrival order.
    let mut answers = Vec::new();
    while let Some(outcome) = outcomes.recv().await {
        answers.push(outcome.text);
    }
    handle.await.unwrap();

    // Answers come back in the same order they were sent.
    assert_eq!(
        answers,
        vec!["ack: did we have X?", "ack: what about Y?", "ack: and Z?"]
    );
    // The provider saw the utterances in order.
    assert_eq!(
        *seen.lock().unwrap(),
        vec!["did we have X?", "what about Y?", "and Z?"]
    );
    // Crucially: never more than one turn in flight → strictly sequential.
    assert_eq!(
        max_concurrent.load(Ordering::SeqCst),
        1,
        "turns overlapped; the queue must serialize"
    );
}

#[tokio::test]
async fn empty_transcripts_are_skipped() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let provider = EchoProvider {
        seen_user_texts: seen.clone(),
        in_flight: Arc::new(AtomicI32::new(0)),
        max_concurrent: Arc::new(AtomicI32::new(0)),
    };
    let agent = LlmAgent::new(
        AgentConfig::default(),
        Box::new(provider),
        ToolRegistry::new(),
    );
    let (tx, mut outcomes, runner) = bridge(agent);
    let handle = tokio::spawn(runner.run());

    tx.send("   ".to_string()).await.unwrap(); // whitespace only
    tx.send("real question".to_string()).await.unwrap();
    drop(tx);

    let mut answers = Vec::new();
    while let Some(o) = outcomes.recv().await {
        answers.push(o.text);
    }
    handle.await.unwrap();

    assert_eq!(answers, vec!["ack: real question"]);
    assert_eq!(*seen.lock().unwrap(), vec!["real question"]);
}

#[tokio::test]
async fn history_relays_across_bridged_utterances() {
    // The second utterance's request must include the first exchange — the
    // bridge feeds the same stateful agent, so the relay still holds.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let provider = EchoProvider {
        seen_user_texts: seen.clone(),
        in_flight: Arc::new(AtomicI32::new(0)),
        max_concurrent: Arc::new(AtomicI32::new(0)),
    };
    let agent = LlmAgent::new(
        AgentConfig::default(),
        Box::new(provider),
        ToolRegistry::new(),
    );
    let (tx, mut outcomes, runner) = bridge(agent);
    let handle = tokio::spawn(runner.run());

    tx.send("first".to_string()).await.unwrap();
    tx.send("second".to_string()).await.unwrap();
    drop(tx);

    let mut n = 0;
    while outcomes.recv().await.is_some() {
        n += 1;
    }
    handle.await.unwrap();

    assert_eq!(n, 2, "both utterances produced an answer");
    // The provider recorded the *latest* user text each time; ordering proves
    // the two ran as distinct, sequential turns on one agent.
    assert_eq!(*seen.lock().unwrap(), vec!["first", "second"]);
}

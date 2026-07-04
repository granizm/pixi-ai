//! Bridge: whisper transcripts (speech-capture) → the agent.
//!
//! `speech-capture` delivers results through a **synchronous** callback running
//! on its VAD worker thread, while [`crate::agent::LlmAgent`] is **async and
//! stateful** (`&mut self`, retained history). You can't `.await` the agent from
//! inside that callback, and several transcripts can arrive back-to-back.
//!
//! This module decouples the two with a channel, exactly as `speech-capture`
//! itself decouples audio capture from VAD:
//!
//! ```text
//! mic → speech-capture (worker thread)
//!         └ callback: VoiceEnd.transcript ──► mpsc ──┐   (sync, non-blocking)
//!                                                     ▼
//!         async runner: rx.recv() → agent.respond().await → outcomes mpsc → caller
//! ```
//!
//! Back-to-back utterances are handled by a **sequential queue**: the runner
//! pulls one transcript at a time and fully resolves it (the whole agentic loop)
//! before taking the next, so ordering and history stay consistent. Coalescing
//! rapid utterances into one input is a later refinement; the queue is the
//! simple, correct baseline.
//!
//! ## Features
//!
//! - `bridge`: the channel + [`SpeechRunner`] (tokio-only; no native deps, so it
//!   builds and tests on any platform). Feed it transcripts from anywhere.
//! - `speech`: adds [`speech_callback`], which adapts a `speech-capture`
//!   `SpeechEvent` stream into the bridge. Pulls speech-capture and its native
//!   ten-vad, so it's a heavier, opt-in feature.

use tokio::sync::mpsc;

use crate::agent::LlmAgent;
use crate::error::AgentError;
use crate::types::AgentOutcome;

/// Default bound on the transcript channel. Whisper utterances arrive at human
/// speaking cadence, so a small buffer is plenty; if the agent falls behind,
/// `try_send` drops the oldest-style overflow (see [`transcript_sender`]).
const DEFAULT_CHANNEL_CAPACITY: usize = 32;

/// Sends finalized transcripts into the bridge. Cloneable and `Send`, so it can
/// be moved into the synchronous `speech-capture` callback.
pub type TranscriptSender = mpsc::Sender<String>;

/// Receives the agent's answers, one per resolved utterance, in order.
pub type OutcomeReceiver = mpsc::Receiver<AgentOutcome>;

/// Construct the bridge channels and the runner future.
///
/// Returns:
/// - a [`TranscriptSender`] to feed transcripts in (wire it to a
///   `speech-capture` callback via [`speech_callback`]),
/// - an [`OutcomeReceiver`] to read answers out,
/// - a `run` future that drives the agent sequentially until the transcript
///   channel closes. Spawn it on your async runtime.
pub fn bridge(agent: LlmAgent) -> (TranscriptSender, OutcomeReceiver, SpeechRunner) {
    let (tx_in, rx_in) = mpsc::channel::<String>(DEFAULT_CHANNEL_CAPACITY);
    let (tx_out, rx_out) = mpsc::channel::<AgentOutcome>(DEFAULT_CHANNEL_CAPACITY);
    let runner = SpeechRunner {
        agent,
        rx_in,
        tx_out,
    };
    (tx_in, rx_out, runner)
}

/// The async side: owns the agent and drains transcripts one at a time.
pub struct SpeechRunner {
    agent: LlmAgent,
    rx_in: mpsc::Receiver<String>,
    tx_out: mpsc::Sender<AgentOutcome>,
}

impl SpeechRunner {
    /// Run until the transcript channel closes (all senders dropped).
    ///
    /// Each transcript is processed to completion before the next is taken —
    /// the sequential queue. A failed turn is logged and skipped so one bad
    /// utterance doesn't tear down the loop; the conversation continues.
    pub async fn run(mut self) {
        while let Some(transcript) = self.rx_in.recv().await {
            if transcript.trim().is_empty() {
                continue;
            }
            let outcome = match self.agent.respond(transcript).await {
                Ok(outcome) => outcome,
                Err(e) => {
                    // Surface the failure to the consumer instead of swallowing
                    // it — otherwise the UI spins on "考え中…" forever. Keep the
                    // loop alive; the next utterance still has prior history.
                    log::warn!("agent failed on a transcript: {e}");
                    AgentOutcome::failure(e.to_string())
                }
            };
            // If the consumer has gone away, stop.
            if self.tx_out.send(outcome).await.is_err() {
                break;
            }
        }
    }

    /// Process exactly one queued transcript, if available, without blocking.
    /// Returns `Ok(None)` if the queue is empty. Useful for poll-style drivers
    /// (e.g. an Android JNI `llmPoll`-type loop) instead of the long-lived
    /// [`run`](Self::run) task.
    pub async fn step(&mut self) -> Result<Option<AgentOutcome>, AgentError> {
        match self.rx_in.try_recv() {
            Ok(transcript) if !transcript.trim().is_empty() => {
                let outcome = self.agent.respond(transcript).await?;
                Ok(Some(outcome))
            }
            Ok(_) => Ok(None), // empty transcript, skip
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Ok(None),
        }
    }
}

/// Build a synchronous `speech-capture` callback that forwards each
/// `VoiceEnd.transcript` into the bridge.
///
/// The returned closure satisfies `FnMut(SpeechEvent) + Send + 'static`, the
/// signature `SpeechCapture::start` expects. It never blocks the VAD worker:
/// it `try_send`s and, if the queue is full (agent far behind), drops that
/// transcript with a warning rather than stalling capture.
///
/// ```ignore
/// let (tx, mut outcomes, runner) = bridge(agent);
/// tokio::spawn(runner.run());
/// speech_capture.start(speech_callback(tx))?;
/// while let Some(answer) = outcomes.recv().await { /* display / TTS */ }
/// ```
#[cfg(feature = "speech")]
pub fn speech_callback(
    sender: TranscriptSender,
) -> impl FnMut(speech_capture::SpeechEvent) + Send + 'static {
    move |event| {
        if let speech_capture::SpeechEvent::VoiceEnd {
            transcript: Some(text),
            ..
        } = event
        {
            if text.trim().is_empty() {
                return;
            }
            match sender.try_send(text) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(dropped)) => {
                    log::warn!("transcript queue full; dropping utterance: {dropped:?}");
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    log::warn!("transcript queue closed; agent runner has stopped");
                }
            }
        }
    }
}

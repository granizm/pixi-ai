//! Voice activity detection from microphone input.
//!
//! Combines `audio-capture` and `ten-vad` (vad) to detect speech segments
//! from microphone input in real time.

pub mod json_log;
pub mod segmenter;
pub mod stt_types;

#[cfg(feature = "stt")]
pub mod whisper_engine;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

pub use audio_capture::{AudioConfig, AudioError};
pub use stt_types::{SttConfig, SttMode};

/// Events emitted during speech capture.
#[derive(Debug, Clone)]
pub enum SpeechEvent {
    /// Voice activity started.
    VoiceStart { timestamp: Duration },
    /// Interim transcription result (streaming mode only).
    TranscriptInterim { timestamp: Duration, text: String },
    /// Voice activity ended with the captured audio.
    VoiceEnd {
        timestamp: Duration,
        audio: Vec<i16>,
        duration: Duration,
        /// Transcription text (only when STT is enabled).
        transcript: Option<String>,
        /// ETD prediction: true = turn complete (only when `end-of-turn` feature is enabled).
        end_of_turn: Option<bool>,
        /// ETD raw probability in [0.0, 1.0] (only when `end-of-turn` feature is enabled).
        turn_probability: Option<f32>,
    },
    /// Per-frame VAD status (only if `emit_vad_status` is enabled).
    VadStatus {
        timestamp: Duration,
        probability: f32,
        is_voice: bool,
    },
}

/// Configuration for speech capture.
#[derive(Debug, Clone)]
pub struct SpeechConfig {
    pub vad_threshold: f32,
    pub hop_size: vad::HopSize,
    /// Ignore utterances shorter than this.
    pub min_speech_duration_ms: u32,
    /// End speech after this much silence.
    pub silence_timeout_ms: u32,
    /// Emit VadStatus events per frame.
    pub emit_vad_status: bool,
    pub audio: AudioConfig,
    /// STT configuration. None = disabled, Some = enable Whisper.
    /// Requires the `stt` feature to actually run transcription.
    pub stt: Option<SttConfig>,
    /// ETD configuration. None = disabled, Some = enable End-of-Turn Detection.
    /// Requires the `end-of-turn` feature.
    #[cfg(feature = "end-of-turn")]
    pub etd: Option<etd::EtdConfig>,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            hop_size: vad::HopSize::Samples256,
            min_speech_duration_ms: 200,
            silence_timeout_ms: 500,
            emit_vad_status: false,
            audio: AudioConfig::default(),
            stt: None,
            #[cfg(feature = "end-of-turn")]
            etd: None,
        }
    }
}

/// Errors from speech capture.
#[derive(Debug, thiserror::Error)]
pub enum SpeechError {
    #[error("audio error: {0}")]
    Audio(#[from] AudioError),
    #[error("vad error: {0}")]
    Vad(#[from] vad::VadError),
    #[error("stt error: {0}")]
    Stt(String),
}

/// Real-time speech capture: microphone -> VAD -> speech events.
pub struct SpeechCapture {
    config: SpeechConfig,
    capture: audio_capture::AudioCapture,
    running: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl SpeechCapture {
    pub fn new(config: SpeechConfig) -> Result<Self, SpeechError> {
        let capture = audio_capture::AudioCapture::new(config.audio.clone())?;
        Ok(Self {
            config,
            capture,
            running: Arc::new(AtomicBool::new(false)),
            worker: None,
        })
    }

    /// Start speech capture. `callback` receives SpeechEvent.
    pub fn start<F>(&mut self, callback: F) -> Result<(), SpeechError>
    where
        F: FnMut(SpeechEvent) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<audio_capture::AudioFrame>();

        // Start audio capture -> send frames to channel
        self.capture.start(move |frame| {
            let _ = tx.send(frame);
        })?;

        let running = self.running.clone();
        running.store(true, Ordering::Relaxed);

        let config = self.config.clone();
        let worker = std::thread::spawn(move || {
            Self::vad_worker(rx, callback, &config, &running);
        });

        self.worker = Some(worker);
        Ok(())
    }

    fn vad_worker<F>(
        rx: mpsc::Receiver<audio_capture::AudioFrame>,
        mut callback: F,
        config: &SpeechConfig,
        running: &AtomicBool,
    ) where
        F: FnMut(SpeechEvent),
    {
        let mut vad_instance = match vad::TenVad::new(config.hop_size, config.vad_threshold) {
            Ok(v) => v,
            Err(e) => {
                log::error!("Failed to create VAD: {e}");
                return;
            }
        };

        let mut seg =
            segmenter::VadSegmenter::new(config.min_speech_duration_ms, config.silence_timeout_ms);

        // Initialize ETD detector if configured and feature is enabled.
        // A single detector is shared between streaming early-cut (segmenter) and
        // batch mode (event loop). Both run on this single worker thread.
        // Arc<Mutex> is used because the closure must be Send for VadSegmenter.
        #[cfg(feature = "end-of-turn")]
        let etd_detector: Option<std::sync::Arc<std::sync::Mutex<etd::EndOfTurnDetector>>> =
            config.etd.as_ref().and_then(|etd_config| {
                match etd::EndOfTurnDetector::new(etd_config.clone()) {
                    Ok(detector) => {
                        log::info!("ETD detector initialized");
                        Some(std::sync::Arc::new(std::sync::Mutex::new(detector)))
                    }
                    Err(e) => {
                        log::error!("Failed to initialize ETD: {e}");
                        None
                    }
                }
            });

        // Wire ETD to segmenter for streaming early-cut mode.
        #[cfg(feature = "end-of-turn")]
        if let Some(ref detector) = etd_detector {
            let detector_for_seg = std::sync::Arc::clone(detector);
            seg.set_etd_predict(Box::new(move |audio: &[i16]| {
                let mut det = detector_for_seg.lock().unwrap();
                match det.predict_i16(audio) {
                    Ok(result) => Some(segmenter::EarlyCutResult {
                        prediction: result.prediction,
                        probability: result.probability,
                    }),
                    Err(e) => {
                        log::warn!("ETD streaming early-cut failed: {e}");
                        None
                    }
                }
            }));
        }

        // Initialize Whisper engine if STT is configured and feature is enabled.
        #[cfg(feature = "stt")]
        let whisper_engine: Option<whisper_engine::WhisperEngine> =
            config.stt.as_ref().and_then(|stt_config| {
                match whisper_engine::WhisperEngine::new(stt_config) {
                    Ok(engine) => {
                        log::info!("Whisper STT engine initialized");
                        Some(engine)
                    }
                    Err(e) => {
                        log::error!("Failed to initialize Whisper STT: {e}");
                        None
                    }
                }
            });

        #[cfg(feature = "stt")]
        let stt_mode: SttMode = config
            .stt
            .as_ref()
            .map(|c| c.mode.clone())
            .unwrap_or(SttMode::Disabled);

        #[cfg(feature = "stt")]
        let mut last_interim_time = Duration::ZERO;

        while running.load(Ordering::Relaxed) {
            let frame = match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(f) => f,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };

            match vad_instance.process(&frame.samples) {
                Ok(result) => {
                    if config.emit_vad_status {
                        callback(SpeechEvent::VadStatus {
                            timestamp: frame.timestamp,
                            probability: result.probability,
                            is_voice: result.is_voice,
                        });
                    }

                    // Streaming interim transcription: emit partial results periodically.
                    #[cfg(feature = "stt")]
                    if let (
                        Some(engine),
                        SttMode::Streaming {
                            interim_interval_ms,
                        },
                    ) = (&whisper_engine, &stt_mode)
                    {
                        if seg.is_speaking() {
                            let elapsed = frame.timestamp.saturating_sub(last_interim_time);
                            if elapsed >= Duration::from_millis(*interim_interval_ms as u64) {
                                let accumulated = seg.accumulated_audio();
                                if !accumulated.is_empty() {
                                    if let Ok(text) = engine.transcribe(accumulated) {
                                        if !text.is_empty() {
                                            callback(SpeechEvent::TranscriptInterim {
                                                timestamp: frame.timestamp,
                                                text,
                                            });
                                        }
                                    }
                                }
                                last_interim_time = frame.timestamp;
                            }
                        }
                    }

                    #[allow(unused_mut)]
                    for mut event in seg.feed(result.is_voice, &frame.samples, frame.timestamp) {
                        // Apply ETD to VoiceEnd events (Batch mode fallback).
                        // Only runs if early-cut didn't already set the fields.
                        #[cfg(feature = "end-of-turn")]
                        if let SpeechEvent::VoiceEnd {
                            ref audio,
                            ref mut end_of_turn,
                            ref mut turn_probability,
                            ..
                        } = event
                        {
                            if end_of_turn.is_none() {
                                if let Some(ref detector) = etd_detector {
                                    match detector.lock().unwrap().predict_i16(audio) {
                                        Ok(etd_result) => {
                                            log::info!(
                                                "ETD: prediction={}, probability={:.4}",
                                                etd_result.prediction,
                                                etd_result.probability
                                            );
                                            *end_of_turn = Some(etd_result.prediction);
                                            *turn_probability = Some(etd_result.probability);
                                        }
                                        Err(e) => {
                                            log::warn!("ETD inference failed: {e}");
                                        }
                                    }
                                }
                            }
                        }

                        // Attach transcription to VoiceEnd when STT is enabled.
                        #[cfg(feature = "stt")]
                        if let SpeechEvent::VoiceEnd {
                            timestamp,
                            audio,
                            duration,
                            end_of_turn,
                            turn_probability,
                            ..
                        } = &event
                        {
                            if let Some(engine) = &whisper_engine {
                                match &stt_mode {
                                    SttMode::Batch | SttMode::Streaming { .. } => {
                                        let stt_start = std::time::Instant::now();
                                        let transcript = engine.transcribe(audio).ok();
                                        let stt_elapsed = stt_start.elapsed();
                                        log::info!(
                                            "STT latency: {:.0}ms (audio: {:.1}s, {:.1}x realtime)",
                                            stt_elapsed.as_millis(),
                                            duration.as_secs_f64(),
                                            stt_elapsed.as_secs_f64() / duration.as_secs_f64(),
                                        );
                                        callback(SpeechEvent::VoiceEnd {
                                            timestamp: *timestamp,
                                            audio: audio.clone(),
                                            duration: *duration,
                                            transcript,
                                            end_of_turn: *end_of_turn,
                                            turn_probability: *turn_probability,
                                        });
                                        // Reset interim timer for next utterance.
                                        #[cfg(feature = "stt")]
                                        {
                                            last_interim_time = Duration::ZERO;
                                        }
                                        continue;
                                    }
                                    SttMode::Disabled => {}
                                }
                            }
                        }

                        callback(event);
                    }
                }
                Err(e) => log::warn!("VAD process error: {e}"),
            }
        }
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        self.capture.stop();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Drop for SpeechCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

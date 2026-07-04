#![cfg(feature = "stt")]

use crate::{SpeechError, SttConfig};

pub struct WhisperEngine {
    ctx: whisper_rs::WhisperContext,
    language: Option<String>,
    /// Directory containing the Whisper model (used to find VAD model)
    model_dir: String,
}

impl WhisperEngine {
    pub fn new(config: &SttConfig) -> Result<Self, SpeechError> {
        let ctx = whisper_rs::WhisperContext::new_with_params(
            &config.model_path,
            whisper_rs::WhisperContextParameters::default(),
        )
        .map_err(|e| SpeechError::Stt(format!("Failed to load Whisper model: {e}")))?;

        let model_dir = std::path::Path::new(&config.model_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        Ok(Self {
            ctx,
            language: config.language.clone(),
            model_dir,
        })
    }

    /// Transcribe audio (i16 16kHz mono) and return full text.
    pub fn transcribe(&self, audio_i16: &[i16]) -> Result<String, SpeechError> {
        let mut audio_f32: Vec<f32> = audio_i16.iter().map(|&s| s as f32 / 32768.0).collect();
        // Loudness-normalize quiet speech so whisper recognizes it better
        // (small/far voices were misrecognized). RMS-of-speech-frames based:
        // robust to isolated spikes and trailing silence — see normalize_loudness.
        // Returns None when the segment is essentially silence (below the energy
        // floor): transcribing it wastes ~13s of CPU and yields hallucinations
        // ("ありがとうございました" etc.), so we skip it entirely.
        if normalize_loudness(&mut audio_f32).is_none() {
            return Ok(String::new());
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| SpeechError::Stt(format!("Failed to create state: {e}")))?;

        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

        if let Some(lang) = &self.language {
            params.set_language(Some(lang));
        }
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Skip timestamp token generation (not needed, saves decoder steps)
        params.set_no_timestamps(true);
        // Single segment output: our VAD already segments speech, so each
        // transcribe() call is one utterance — no need for internal re-segmentation.
        params.set_single_segment(true);
        // Suppress non-speech tokens (music notes, applause markers, etc.)
        // Prevents decoder from wasting steps on non-speech output.
        params.set_suppress_nst(true);

        // ── Noise-robust inference parameters ──
        //
        // Anti-hallucination: discard segments with high entropy (low confidence)
        params.set_entropy_thold(2.4);
        // Prevent context-carry loops: noise in one segment propagates errors
        // to subsequent segments via conditioning. Disabling this is critical
        // for noisy environments (proven to reduce hallucination cascades).
        params.set_no_context(true);
        // Lower no-speech threshold: more aggressively skip noise-only segments.
        // Default 0.6 lets some noise through; 0.4 filters more aggressively.
        params.set_no_speech_thold(0.4);
        // Deterministic decoding (no temperature sampling)
        params.set_temperature(0.0);
        // Disable temperature fallback (avoid slow retries on noisy input)
        params.set_temperature_inc(0.0);
        // NOTE: initial_prompt was removed (was: "これは日本語の音声です。").
        // On low-confidence audio the prompt leaks into the output as a
        // continuation ("この音声は、…" / "その他の音声は、…" — observed on
        // device 2026-07-04). Language is already forced via set_language,
        // so the prompt added nothing but the leak.

        // ── Silero VAD (whisper.cpp built-in) ──
        //
        // If a Silero VAD model is found next to the Whisper model,
        // enable it to skip silence within the audio segment.
        // This is a second layer of VAD (after TenVad segmentation):
        // TenVad decides WHEN to send audio, Silero decides WHAT to skip inside it.
        if let Some(vad_path) = self.find_vad_model() {
            log::info!("Enabling whisper.cpp built-in Silero VAD: {}", vad_path);
            params.set_vad_model_path(Some(&vad_path));
        }

        state
            .full(params, &audio_f32)
            .map_err(|e| SpeechError::Stt(format!("Whisper transcription failed: {e}")))?;

        let mut text = String::new();
        let n_segments = state.full_n_segments();
        for i in 0..n_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str() {
                    text.push_str(segment_text);
                }
            }
        }

        let text = text.trim();
        if is_hallucination(text) {
            log::info!("whisper: dropped hallucination transcript: \"{text}\"");
            return Ok(String::new());
        }
        Ok(text.to_string())
    }

    /// Look for a Silero VAD model in the same directory as the Whisper model.
    fn find_vad_model(&self) -> Option<String> {
        let candidates = [
            "ggml-silero-v6.2.0.bin",
            "ggml-silero-v5.1.2.bin",
            "silero-vad.onnx",
        ];
        for name in &candidates {
            let path = std::path::Path::new(&self.model_dir).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().to_string());
            }
        }
        None
    }
}

/// Loudness-normalize a mono f32 [-1,1] speech segment in place.
///
/// v1 was peak normalization, which had an inconsistency failure mode: the
/// gain is decided by the single loudest sample, so one click / plosive /
/// mic bump anywhere in the segment blocks the boost entirely, while
/// spike-free segments get boosted ~10x — recognition quality then swings
/// wildly between utterances ("effective sometimes, useless other times").
///
/// Instead, estimate the *speech* RMS from the loudest 40% of 20ms frames —
/// robust to both trailing silence (sits in the quiet percentiles) and
/// isolated spikes (one loud frame is diluted by the rest) — and normalize
/// that to a target level. Rare samples that exceed full scale after the
/// gain are hard-clipped, which is harmless for ASR.
/// Returns `Some(gain)` when the segment contains usable speech (gain may be
/// 1.0 when no boost was needed), or `None` when the segment is below the
/// energy floor — i.e. essentially silence/noise that should not be
/// transcribed at all. The floor assumes AGC-leveled input
/// (AudioSource.VOICE_COMMUNICATION): real speech measures ≳0.018 RMS there,
/// while silence segments measure ~0.0001 — a >100x separation.
fn normalize_loudness(samples: &mut [f32]) -> Option<f32> {
    const FRAME: usize = 320; // 20ms @ 16kHz
    const TARGET_RMS: f32 = 0.1; // ~ -20 dBFS, comfortable speech level
    const MAX_GAIN: f32 = 30.0;
    const MIN_SPEECH_RMS: f32 = 0.0025; // energy floor: below this = silence

    if samples.len() < FRAME {
        return Some(1.0);
    }

    // Per-frame RMS.
    let mut frame_rms: Vec<f32> = samples
        .chunks(FRAME)
        .map(|c| (c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();

    // Speech RMS ≈ mean of the loudest 40% of frames.
    frame_rms.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let take = (frame_rms.len() * 2 / 5).max(1);
    let speech_rms = frame_rms[..take].iter().sum::<f32>() / take as f32;

    if speech_rms < MIN_SPEECH_RMS {
        log::info!(
            "whisper: segment below energy floor (speech_rms={speech_rms:.4}), skipping as silence"
        );
        return None;
    }
    let mut gain = TARGET_RMS / speech_rms;
    if gain <= 1.0 {
        log::info!("whisper: loudness ok (speech_rms={speech_rms:.4}), no gain");
        return Some(1.0);
    }
    if gain > MAX_GAIN {
        gain = MAX_GAIN;
    }
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    log::info!("whisper: loudness-normalized (speech_rms={speech_rms:.4}, gain={gain:.2})");
    Some(gain)
}

/// Known Whisper-Japanese hallucination phrases (YouTube-caption training
/// artifacts) that surface on noise/silence-only segments. A short transcript
/// that is essentially one of these is discarded.
fn is_hallucination(text: &str) -> bool {
    const PHRASES: [&str; 4] = [
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "チャンネル登録",
        "最後までご視聴いただき",
    ];
    // Long sentences are real speech even if they contain one of the phrases.
    if text.chars().count() > 30 {
        return false;
    }
    PHRASES.iter().any(|p| text.contains(p))
}

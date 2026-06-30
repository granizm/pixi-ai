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
        // Peak-normalize quiet speech so whisper recognizes it better (small/far
        // voices were misrecognized). Scale so the loudest sample hits ~0.9, with
        // a gain cap so near-silence/noise isn't blown up. Matches the on-device
        // (Gemma 4) path's normalization for consistent behavior across engines.
        normalize_peak(&mut audio_f32);

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
        // Style hint for Japanese transcription output formatting
        params.set_initial_prompt("これは日本語の音声です。");

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

        Ok(text.trim().to_string())
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

/// Peak-normalize a mono f32 [-1,1] buffer in place: scale so the loudest sample
/// reaches ~0.9 of full scale, capped so near-silence/noise isn't amplified into
/// a roar. No-op if already loud enough or if the buffer is silent.
fn normalize_peak(samples: &mut [f32]) {
    const TARGET_PEAK: f32 = 0.9;
    const MAX_GAIN: f32 = 12.0;

    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak <= f32::EPSILON {
        return; // silence
    }
    let mut gain = TARGET_PEAK / peak;
    if gain <= 1.0 {
        return; // already loud enough — don't attenuate
    }
    if gain > MAX_GAIN {
        gain = MAX_GAIN;
    }
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
    log::debug!("whisper: peak-normalized (peak={peak:.3}, gain={gain:.2})");
}

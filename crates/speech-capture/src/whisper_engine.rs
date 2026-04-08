#![cfg(feature = "stt")]

use crate::{SpeechError, SttConfig};

pub struct WhisperEngine {
    ctx: whisper_rs::WhisperContext,
    language: Option<String>,
}

impl WhisperEngine {
    pub fn new(config: &SttConfig) -> Result<Self, SpeechError> {
        let ctx = whisper_rs::WhisperContext::new_with_params(
            &config.model_path,
            whisper_rs::WhisperContextParameters::default(),
        )
        .map_err(|e| SpeechError::Stt(format!("Failed to load Whisper model: {e}")))?;

        Ok(Self {
            ctx,
            language: config.language.clone(),
        })
    }

    /// Transcribe audio (i16 16kHz mono) and return full text.
    pub fn transcribe(&self, audio_i16: &[i16]) -> Result<String, SpeechError> {
        let audio_f32: Vec<f32> = audio_i16.iter().map(|&s| s as f32 / 32768.0).collect();

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
        params.set_single_segment(false);

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

        // ── Encoder context optimization ──
        //
        // Whisper pads all input to 30s (1500 mel frames) by default.
        // Set audio_ctx to match actual audio length so the encoder skips
        // zero-padded frames. This directly reduces encoder computation.
        // 1 mel frame = 20ms, so: audio_ctx = ceil(duration_secs * 50)
        let duration_secs = audio_f32.len() as f32 / 16000.0;
        let audio_ctx = ((duration_secs * 50.0).ceil() as i32).min(1500).max(1);
        params.set_audio_ctx(audio_ctx);

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
}

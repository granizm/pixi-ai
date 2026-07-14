//! Platform 非依存の non-fatal recorder bridge。
//!
//! Rust 層のイベント (panic / 将来の ggml abort 等) を、ホスト側 (Swift / JNI) が
//! 登録した callback 経由で crash reporter (Crashlytics non-fatal) に届ける。
//!
//! - `set_recorder`: ホストが `extern "C" fn(*const c_char)` を 1 本登録する
//!   (iOS: Swift @convention(c) → Crashlytics.record(error:) /
//!    Android: JNI 側 fn → FirebaseCrashlytics.recordException)。
//! - `record`: 任意のメッセージを recorder へ送る (未登録なら no-op)。
//! - `install_panic_hook`: panic hook を **chain** で仕込む (置換しない)。
//!   既存 hook (log-panics 等) を prev として保持し、record 後に prev を呼ぶ。
//!
//! # 順序の要件 (重要)
//! `log_panics::init()` は hook を**置換**するため、`install_panic_hook` は
//! 必ず log-panics の後に呼ぶこと。アプリの FFI 層は「init_logger() →
//! install_panic_hook()」の順を守る (rust_set_nonfatal_recorder 実装参照)。

use std::os::raw::c_char;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Once;

/// ホストが登録する recorder callback の型。
/// msg は NUL 終端 UTF-8。callback は呼出し中のみ msg を参照してよい
/// (保持するならコピーすること)。
pub type RecorderFn = extern "C" fn(msg: *const c_char);

static RECORDER: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// recorder を登録する (上書き可)。
pub fn set_recorder(cb: RecorderFn) {
    RECORDER.store(cb as *mut (), Ordering::SeqCst);
}

/// メッセージを recorder へ送る。未登録なら no-op。
/// panic hook 内から呼ばれるため、この関数自身は panic しない
/// (CString 失敗時は NUL を除去して再試行し、それでも駄目なら諦める)。
pub fn record(msg: &str) {
    let p = RECORDER.load(Ordering::SeqCst);
    if p.is_null() {
        return;
    }
    // SAFETY: RECORDER には set_recorder で登録された RecorderFn しか入らない。
    let cb: RecorderFn = unsafe { std::mem::transmute::<*mut (), RecorderFn>(p) };
    let sanitized;
    let bytes = if msg.as_bytes().contains(&0) {
        sanitized = msg.replace('\0', " ");
        sanitized.as_bytes()
    } else {
        msg.as_bytes()
    };
    if let Ok(cstr) = std::ffi::CString::new(bytes) {
        cb(cstr.as_ptr());
    }
}

/// panic メッセージの最大長 (backtrace 込み)。Crashlytics 側の負担と
/// JNI/NSError の実用性から 16KB に制限する。
const MAX_MSG_LEN: usize = 16 * 1024;

/// panic hook を chain で仕込む (冪等)。
/// 既存 hook (log-panics 等) を prev として保持し、record → prev の順で呼ぶ。
pub fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // RUST_BACKTRACE 環境変数に依存せず必ず取得する
            let bt = std::backtrace::Backtrace::force_capture();
            let mut msg = format!("rust panic: {info}\nbacktrace:\n{bt}");
            if msg.len() > MAX_MSG_LEN {
                // char 境界を守って切る
                let mut end = MAX_MSG_LEN;
                while !msg.is_char_boundary(end) {
                    end -= 1;
                }
                msg.truncate(end);
                msg.push_str("\n…(truncated)");
            }
            record(&msg);
            prev(info);
        }));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    extern "C" fn test_recorder(msg: *const c_char) {
        assert!(!msg.is_null());
        let s = unsafe { std::ffi::CStr::from_ptr(msg) }.to_str().unwrap();
        assert!(s.contains("rust panic") || s.contains("direct"));
        CALLS.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn record_is_noop_without_recorder() {
        // 未登録でも落ちない (他テストとの実行順で recorder が入り得るため
        // カウントは検証しない)
        record("direct message with no recorder");
    }

    #[test]
    fn panic_hook_records_and_chains() {
        set_recorder(test_recorder);
        install_panic_hook();
        let before = CALLS.load(Ordering::SeqCst);
        let _ = std::panic::catch_unwind(|| panic!("test panic for hook"));
        assert!(CALLS.load(Ordering::SeqCst) > before);
        // NUL 入りメッセージも落ちない
        record("direct\0with nul");
    }
}

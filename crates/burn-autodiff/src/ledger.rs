//! Opt-in autodiff retention ledger (loractl#132, ADR-0005 attribution).
//!
//! When the `LORACTL_RETENTION_LEDGER` environment variable names a writable
//! file path, burn-autodiff appends one tab-separated line per
//! retention-relevant event:
//!
//! - `OP\t<node>\t<op_type>\t<parent nodes>` — a tracked op finished; maps the
//!   node id to the op's `Backward` type name so later events can be classed.
//! - `CKPT\t<node>\t<Explicit|Backup>\t<Computed|Recompute>\t<bytes>\t<shape>`
//!   — a checkpoint action was registered during forward. A `Computed` action
//!   clones the primitive (an Arc'd handle) **at this moment**, pinning the
//!   underlying buffer until backward; `Recompute` stores only a retro-forward
//!   (bytes logged as 0).
//! - `FALLBACK\t<op_type>` — a memory-bound op was flipped to ComputeBound
//!   because one of its parents is untracked
//!   (`CheckpointingError::UntrackedParent`).
//! - `BUILD\t<node>\t<Computed|Recompute>\t<n_required>` — the state map
//!   insert when the checkpointer is built at backward start. Actions that were
//!   registered but never required are logged as `DROP\t<node>` instead (their
//!   pinned clone is released here, having been held for the whole forward).
//! - `SAVE\t<node>\t<n_required>` — a recomputed output was materialized into
//!   the state map during backward (retro-forward execution).
//! - `CONSUME\t<node>\t<n_remaining>` — a backward step consumed the state;
//!   `0` means the entry was freed.
//!
//! Callers outside this crate (the trainer / probe binary) may append
//! `PHASE\t<name>` marker lines to the same file to segment the timeline.
//!
//! Lines are flushed per event on purpose: the consumer is an OOM
//! investigation, and an aborting process must not lose the tail — the tail IS
//! the measurement.
//!
//! This is instrumentation only: no tensors are cloned, copied, or
//! synchronized here — the ledger observes host-side graph bookkeeping and
//! cannot perturb the retention it measures. With the env var unset the cost
//! is one lazily-initialized `Option` check per event site.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

static LEDGER: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();

fn writer() -> Option<&'static Mutex<BufWriter<File>>> {
    LEDGER
        .get_or_init(|| {
            std::env::var_os("LORACTL_RETENTION_LEDGER").map(|path| {
                let file = File::options()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .unwrap_or_else(|e| {
                        panic!("LORACTL_RETENTION_LEDGER={path:?} must be writable: {e}")
                    });
                Mutex::new(BufWriter::new(file))
            })
        })
        .as_ref()
}

/// Whether the ledger is active (the env var was set at first use).
pub fn enabled() -> bool {
    writer().is_some()
}

/// Append one event line. No-op when the ledger is inactive.
pub fn log(line: core::fmt::Arguments<'_>) {
    if let Some(mutex) = writer() {
        let mut w = mutex.lock().unwrap();
        let _ = writeln!(w, "{line}");
        // Flush per line: an OOM abort must not lose the tail.
        let _ = w.flush();
    }
}

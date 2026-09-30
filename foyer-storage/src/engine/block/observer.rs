// Copyright 2026 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::fmt::Debug;

/// Why an entry left the block engine index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DepartureReason {
    /// Recovery found a record superseded by a newer record or a tombstone and did not restore it.
    RecoveryDiscard,
    /// A read found a checksum mismatch or a malformed entry header and removed the entry.
    ReadCorruption,
    /// The block holding the entry was reclaimed without reinserting the entry.
    Reclaim,
    /// A newer insertion of the same key replaced the entry.
    Replacement,
    /// The entry was deleted or the disk cache was destroyed.
    ExplicitDelete,
}

/// One entry leaving the block engine index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Departure {
    /// Hash of the entry key.
    pub hash: u64,
    /// Serialized key and value bytes of the entry, excluding its header and alignment padding.
    pub payload_bytes: usize,
    /// Why the entry left.
    pub reason: DepartureReason,
}

/// Why the block engine dropped a write before it reached the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WriteDropReason {
    /// An unpaced write found the submit queue past its threshold.
    QueueFull,
    /// The entry does not fit an empty flush buffer.
    Oversized,
    /// The engine was closing.
    Closed,
}

/// A write the block engine dropped before it reached the disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroppedWrite {
    /// Hash of the entry key.
    pub hash: u64,
    /// Why the write was dropped.
    pub reason: WriteDropReason,
}

/// Outcome of a successful block engine recovery.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Entries restored into the index.
    pub restored_entries: usize,
    /// Serialized key and value bytes of the restored entries.
    pub restored_payload_bytes: usize,
    /// Records found on disk but not restored, each also reported as a
    /// [`DepartureReason::RecoveryDiscard`] departure.
    pub discarded_records: usize,
    /// Blocks whose scan stopped at a read error under [`crate::RecoverMode::Quiet`]; their remaining records were
    /// not recovered, so the restore is incomplete. A blob whose index fails its checksum ends the block's scan like
    /// unwritten space and is not counted here.
    pub corrupt_blocks: usize,
}

/// Observes the entry lifecycle of the block engine.
///
/// Callbacks run on the engine's IO and caller paths after the index is
/// updated, so implementations must be cheap and must not block.
pub trait EntryObserver: Send + Sync + Debug + 'static {
    /// Called once for every entry that leaves the index.
    fn on_departure(&self, departure: Departure);

    /// Called once when recovery finishes successfully.
    fn on_recovery(&self, report: RecoveryReport);

    /// Called once for every write dropped before it reached the disk.
    fn on_dropped_write(&self, dropped: DroppedWrite);
}

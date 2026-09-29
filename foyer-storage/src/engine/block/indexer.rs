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

use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use itertools::Itertools;
use parking_lot::RwLock;

use crate::engine::block::{
    manager::BlockId,
    observer::{Departure, DepartureReason, EntryObserver},
    serde::{EntryHeader, Sequence},
};

#[derive(Debug, Clone)]
pub enum Index {
    Address(EntryAddress),
    Tombstone(Sequence),
}

impl Index {
    fn sequence(&self) -> Sequence {
        match self {
            Index::Address(addr) => addr.sequence,
            Index::Tombstone(seq) => *seq,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HashedEntryAddress {
    pub hash: u64,
    pub address: EntryAddress,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryAddress {
    pub block: BlockId,
    pub offset: u32,
    pub len: u32,

    pub sequence: Sequence,
}

impl EntryAddress {
    /// Serialized key and value bytes, excluding the entry header and alignment padding.
    pub fn payload(&self) -> usize {
        self.len as usize - EntryHeader::serialized_len()
    }
}

type IndexerShard = HashMap<u64, Index>;

/// [`Indexer`] records key hash to entry address on fs.
///
/// It is the only owner of the live entry set, so it also maintains the live payload bytes and reports every
/// departure. The payload counter is updated under the shard lock that changes the entry, which keeps it equal to the
/// sum of the indexed payloads whenever no update is in progress.
#[derive(Debug, Clone)]
pub struct Indexer {
    shards: Arc<Vec<RwLock<IndexerShard>>>,
    payload: Arc<AtomicUsize>,
    observer: Option<Arc<dyn EntryObserver>>,
}

impl Indexer {
    pub fn new(shards: usize, observer: Option<Arc<dyn EntryObserver>>) -> Self {
        let shards = (0..shards).map(|_| RwLock::new(HashMap::new())).collect_vec();
        Self {
            shards: Arc::new(shards),
            payload: Arc::default(),
            observer,
        }
    }

    /// Serialized key and value bytes of all indexed entries.
    pub fn payload_bytes(&self) -> usize {
        self.payload.load(Ordering::Relaxed)
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::insert_tombstone")
    )]
    pub fn insert_tombstone(&self, hash: u64, sequence: Sequence) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        let mut departures = vec![];
        let old = {
            let mut shard = self.shards[shard].write();
            self.insert_inner(&mut shard, hash, Index::Tombstone(sequence), &mut departures)
        };
        self.report(departures);
        old
    }

    /// Returns the addresses the batch invalidated: replaced or relocated addresses, and inserted addresses rejected
    /// because a newer sequence is already indexed.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::insert_batch")
    )]
    pub fn insert_batch(&self, batch: Vec<HashedEntryAddress>) -> Vec<HashedEntryAddress> {
        let shards: HashMap<usize, Vec<HashedEntryAddress>> =
            batch.into_iter().into_group_map_by(|haddr| self.shard(haddr.hash));

        let mut olds = vec![];
        let mut departures = vec![];
        for (s, batch) in shards {
            let mut shard = self.shards[s].write();
            for haddr in batch {
                if let Some(old) =
                    self.insert_inner(&mut shard, haddr.hash, Index::Address(haddr.address), &mut departures)
                {
                    olds.push(HashedEntryAddress {
                        hash: haddr.hash,
                        address: old,
                    });
                }
            }
        }
        self.report(departures);
        olds
    }

    #[cfg_attr(feature = "tracing", fastrace::trace(name = "foyer::storage::block::indexer::get"))]
    pub fn get(&self, hash: u64) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        match self.shards[shard].read().get(&hash) {
            Some(index) => match index {
                Index::Address(addr) => Some(addr.clone()),
                Index::Tombstone(_) => None,
            },
            None => None,
        }
    }

    /// Removes the entry only while it is still indexed at `address`, so a stale observation cannot remove a newer
    /// replacement or relocation.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::remove")
    )]
    pub fn remove(&self, hash: u64, address: &EntryAddress, reason: DepartureReason) -> bool {
        let removed = self.remove_addresses([(hash, address)], reason);
        removed == 1
    }

    /// Removes each entry only while it is still indexed at the given address; returns how many were removed.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::remove_addresses")
    )]
    pub fn remove_addresses<'a, I>(&self, batch: I, reason: DepartureReason) -> usize
    where
        I: IntoIterator<Item = (u64, &'a EntryAddress)>,
    {
        let shards = batch.into_iter().into_group_map_by(|(hash, _)| self.shard(*hash));

        let mut departures = vec![];
        for (s, addresses) in shards {
            let mut shard = self.shards[s].write();
            for (hash, address) in addresses {
                if let Entry::Occupied(o) = shard.entry(hash)
                    && matches!(o.get(), Index::Address(current) if current == address)
                {
                    o.remove();
                    self.depart(hash, address, reason, &mut departures);
                }
            }
        }
        let removed = departures.len();
        self.report(departures);
        removed
    }

    /// Removes the indexed tombstones or entries not newer than the given sequences.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::remove_batch")
    )]
    pub fn remove_batch<I>(&self, batch: I) -> Vec<EntryAddress>
    where
        I: IntoIterator<Item = (u64, Sequence)>,
    {
        let shards = batch.into_iter().into_group_map_by(|(hash, _)| self.shard(*hash));

        let mut olds = vec![];
        let mut departures = vec![];
        for (s, hashes) in shards {
            let mut shard = self.shards[s].write();
            for (hash, sequence) in hashes {
                match shard.entry(hash) {
                    Entry::Occupied(o) => {
                        if sequence >= o.get().sequence()
                            && let Some(addr) = self.extract_address(o.remove())
                        {
                            self.depart(hash, &addr, DepartureReason::ExplicitDelete, &mut departures);
                            olds.push(addr);
                        }
                    }
                    Entry::Vacant(_) => {}
                }
            }
        }
        self.report(departures);
        olds
    }

    #[cfg_attr(feature = "tracing", fastrace::trace(name = "foyer::storage::block::indexer::clear"))]
    pub fn clear(&self) {
        let mut departures = vec![];
        for shard in self.shards.iter() {
            let mut shard = shard.write();
            for (hash, index) in shard.drain() {
                if let Index::Address(addr) = index {
                    self.depart(hash, &addr, DepartureReason::ExplicitDelete, &mut departures);
                }
            }
        }
        self.report(departures);
    }

    #[inline(always)]
    fn shard(&self, hash: u64) -> usize {
        hash as usize % self.shards.len()
    }

    fn insert_inner(
        &self,
        shard: &mut IndexerShard,
        hash: u64,
        index: Index,
        departures: &mut Vec<Departure>,
    ) -> Option<EntryAddress> {
        match shard.entry(hash) {
            Entry::Occupied(mut o) => {
                // `>` for updates.
                // '=' for reinsertions.
                if index.sequence() >= o.get().sequence() {
                    let reason = match &index {
                        Index::Tombstone(_) => Some(DepartureReason::ExplicitDelete),
                        Index::Address(_) if index.sequence() > o.get().sequence() => {
                            Some(DepartureReason::Replacement)
                        }
                        // A reinsertion relocates the same entry, so nothing departs.
                        Index::Address(_) => None,
                    };
                    self.arrive(&index);
                    let old = self.extract_address(o.insert(index));
                    if let Some(old) = &old {
                        match reason {
                            Some(reason) => self.depart(hash, old, reason, departures),
                            None => {
                                self.payload.fetch_sub(old.payload(), Ordering::Relaxed);
                            }
                        }
                    }
                    old
                } else {
                    self.extract_address(index)
                }
            }
            Entry::Vacant(v) => {
                self.arrive(&index);
                v.insert(index);
                None
            }
        }
    }

    fn arrive(&self, index: &Index) {
        if let Index::Address(addr) = index {
            self.payload.fetch_add(addr.payload(), Ordering::Relaxed);
        }
    }

    fn depart(&self, hash: u64, address: &EntryAddress, reason: DepartureReason, departures: &mut Vec<Departure>) {
        let payload_bytes = address.payload();
        self.payload.fetch_sub(payload_bytes, Ordering::Relaxed);
        if self.observer.is_some() {
            departures.push(Departure {
                hash,
                payload_bytes,
                reason,
            });
        }
    }

    fn report(&self, departures: Vec<Departure>) {
        if let Some(observer) = &self.observer {
            departures
                .into_iter()
                .for_each(|departure| observer.on_departure(departure));
        }
    }

    fn extract_address(&self, index: Index) -> Option<EntryAddress> {
        match index {
            Index::Address(addr) => Some(addr),
            Index::Tombstone(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn indexed_payload_bytes(&self) -> usize {
        self.shards
            .iter()
            .flat_map(|shard| {
                shard
                    .read()
                    .values()
                    .filter_map(|index| match index {
                        Index::Address(addr) => Some(addr.payload()),
                        Index::Tombstone(_) => None,
                    })
                    .collect_vec()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;
    use crate::engine::block::observer::RecoveryReport;

    const HEADER: usize = EntryHeader::serialized_len();

    #[derive(Debug, Default)]
    struct Recorder(Mutex<Vec<Departure>>);

    impl EntryObserver for Recorder {
        fn on_departure(&self, departure: Departure) {
            self.0.lock().push(departure);
        }

        fn on_recovery(&self, _: RecoveryReport) {}
    }

    impl Recorder {
        fn take(&self) -> Vec<Departure> {
            std::mem::take(&mut *self.0.lock())
        }
    }

    fn indexer() -> (Indexer, Arc<Recorder>) {
        let recorder = Arc::new(Recorder::default());
        (Indexer::new(4, Some(recorder.clone())), recorder)
    }

    fn address(block: BlockId, payload: usize, sequence: Sequence) -> EntryAddress {
        EntryAddress {
            block,
            offset: 0,
            len: (HEADER + payload) as u32,
            sequence,
        }
    }

    fn insert(indexer: &Indexer, hash: u64, address: EntryAddress) -> Vec<HashedEntryAddress> {
        indexer.insert_batch(vec![HashedEntryAddress { hash, address }])
    }

    fn departure(hash: u64, payload_bytes: usize, reason: DepartureReason) -> Departure {
        Departure {
            hash,
            payload_bytes,
            reason,
        }
    }

    #[test]
    fn test_insert_counts_payload_without_header() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 100, 1));
        insert(&indexer, 2, address(0, 50, 2));
        assert_eq!(indexer.payload_bytes(), 150);
        assert_eq!(indexer.payload_bytes(), indexer.indexed_payload_bytes());
        assert!(recorder.take().is_empty());
    }

    #[test]
    fn test_replacement_swaps_payload_once() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 100, 1));
        insert(&indexer, 1, address(1, 30, 2));
        assert_eq!(indexer.payload_bytes(), 30);
        assert_eq!(recorder.take(), vec![departure(1, 100, DepartureReason::Replacement)]);
    }

    #[test]
    fn test_older_sequence_is_rejected_without_change() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 100, 5));
        let rejected = insert(&indexer, 1, address(1, 30, 4));
        assert_eq!(rejected[0].address, address(1, 30, 4));
        assert_eq!(indexer.get(1), Some(address(0, 100, 5)));
        assert_eq!(indexer.payload_bytes(), 100);
        assert!(recorder.take().is_empty());
    }

    #[test]
    fn test_same_sequence_relocation_is_not_a_departure() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 100, 1));
        let relocated = insert(&indexer, 1, address(3, 100, 1));
        assert_eq!(relocated[0].address, address(0, 100, 1));
        assert_eq!(indexer.get(1), Some(address(3, 100, 1)));
        assert_eq!(indexer.payload_bytes(), 100);
        assert!(recorder.take().is_empty());
    }

    #[test]
    fn test_tombstone_removes_payload_and_repeats_without_underflow() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 100, 1));
        assert_eq!(indexer.insert_tombstone(1, 2), Some(address(0, 100, 1)));
        assert_eq!(indexer.insert_tombstone(1, 3), None);
        assert_eq!(indexer.insert_tombstone(9, 4), None);
        assert_eq!(indexer.payload_bytes(), 0);
        assert_eq!(
            recorder.take(),
            vec![departure(1, 100, DepartureReason::ExplicitDelete)]
        );

        indexer.remove_batch([(1, 3), (9, 4)]);
        assert_eq!(indexer.payload_bytes(), 0);
        assert!(recorder.take().is_empty());
    }

    #[test]
    fn test_stale_read_corruption_keeps_newer_copy() {
        let (indexer, recorder) = indexer();
        let stale = address(0, 100, 1);
        insert(&indexer, 1, stale.clone());
        insert(&indexer, 1, address(1, 100, 1));
        assert!(!indexer.remove(1, &stale, DepartureReason::ReadCorruption));
        insert(&indexer, 1, address(2, 70, 2));
        assert!(!indexer.remove(1, &stale, DepartureReason::ReadCorruption));
        assert_eq!(indexer.get(1), Some(address(2, 70, 2)));
        assert_eq!(indexer.payload_bytes(), 70);
        assert_eq!(recorder.take(), vec![departure(1, 100, DepartureReason::Replacement)]);

        assert!(indexer.remove(1, &address(2, 70, 2), DepartureReason::ReadCorruption));
        assert_eq!(indexer.payload_bytes(), 0);
        assert_eq!(recorder.take(), vec![departure(1, 70, DepartureReason::ReadCorruption)]);
    }

    #[test]
    fn test_reclaim_removes_only_its_addresses() {
        let (indexer, recorder) = indexer();
        let reclaimed = [address(0, 10, 1), address(0, 20, 2), address(0, 30, 3)];
        for (hash, address) in reclaimed.iter().enumerate() {
            insert(&indexer, hash as u64, address.clone());
        }
        insert(&indexer, 1, address(1, 25, 4));
        indexer.insert_tombstone(2, 5);

        let removed = indexer.remove_addresses(
            reclaimed
                .iter()
                .enumerate()
                .map(|(hash, address)| (hash as u64, address)),
            DepartureReason::Reclaim,
        );
        assert_eq!(removed, 1);
        assert_eq!(indexer.get(1), Some(address(1, 25, 4)));
        assert_eq!(indexer.payload_bytes(), 25);
        assert_eq!(indexer.payload_bytes(), indexer.indexed_payload_bytes());
        let departures = recorder.take();
        assert_eq!(
            departures,
            vec![
                departure(1, 20, DepartureReason::Replacement),
                departure(2, 30, DepartureReason::ExplicitDelete),
                departure(0, 10, DepartureReason::Reclaim),
            ]
        );
    }

    #[test]
    fn test_clear_reports_every_entry() {
        let (indexer, recorder) = indexer();
        insert(&indexer, 1, address(0, 10, 1));
        insert(&indexer, 2, address(0, 20, 2));
        indexer.insert_tombstone(3, 3);
        indexer.clear();
        assert_eq!(indexer.payload_bytes(), 0);
        let mut departures = recorder.take();
        departures.sort_by_key(|departure| departure.hash);
        assert_eq!(
            departures,
            vec![
                departure(1, 10, DepartureReason::ExplicitDelete),
                departure(2, 20, DepartureReason::ExplicitDelete),
            ]
        );
    }

    #[test]
    fn test_concurrent_shards_match_index() {
        let (indexer, recorder) = indexer();
        let threads = (0..8u64)
            .map(|thread| {
                let indexer = indexer.clone();
                std::thread::spawn(move || {
                    for round in 0..1000u64 {
                        let hash = (thread * 7 + round) % 64;
                        let sequence = thread * 10_000 + round + 1;
                        let payload = ((thread + round) % 97 + 1) as usize;
                        match round % 4 {
                            0 | 1 => {
                                insert(&indexer, hash, address(thread as BlockId, payload, sequence));
                            }
                            2 => {
                                indexer.insert_tombstone(hash, sequence);
                            }
                            _ => {
                                if let Some(current) = indexer.get(hash) {
                                    indexer.remove(hash, &current, DepartureReason::ReadCorruption);
                                }
                            }
                        }
                    }
                })
            })
            .collect_vec();
        threads.into_iter().for_each(|thread| thread.join().unwrap());

        assert_eq!(indexer.payload_bytes(), indexer.indexed_payload_bytes());
        let departures = recorder.take();
        assert!(!departures.is_empty());
        indexer.clear();
        assert_eq!(indexer.payload_bytes(), 0);
    }
}

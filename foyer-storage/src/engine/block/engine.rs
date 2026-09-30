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
    fmt::Debug,
    future::Future,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use asyncband::mpsc::UnboundedReceiver;
#[cfg(feature = "tracing")]
use fastrace::prelude::*;
use foyer_common::{
    bits,
    code::{StorageKey, StorageValue},
    error::{Error, ErrorKind, Result},
    metrics::Metrics,
    properties::{Age, Properties},
    spawn::Spawner,
};
use futures_core::future::BoxFuture;
use futures_util::{
    FutureExt,
    future::{join_all, try_join_all},
};
use itertools::Itertools;

use super::{
    flusher::{Flusher, InvalidStats, Submission},
    indexer::Indexer,
    recover::RecoverRunner,
};
#[cfg(any(test, feature = "test_utils"))]
use crate::test_utils::*;
use crate::{
    Device, Load, RejectAll, StorageFilter, StorageFilterResult,
    compress::Compression,
    engine::{
        Engine, EngineBuildContext, EngineConfig, Populated,
        block::{
            eviction::{EvictionPicker, FifoPicker, InvalidRatioPicker},
            manager::{BlockId, BlockManager},
            observer::{DepartureReason, EntryObserver, WriteDropReason},
            queue::{Reservation, SubmitQueue},
            reclaimer::{BlockCleaner, Reclaimer, ReclaimerTrait},
            serde::{AtomicSequence, EntryHeader},
            tombstone::{Tombstone, TombstoneLog},
        },
    },
    filter::conditions::IoThrottle,
    io::{PAGE, bytes::IoSliceMut},
    keeper::PieceRef,
    serde::EntryDeserializer,
};

/// Config for the block-based disk cache engine.
///
/// The block-based disk cache engine is suitable for general cache entries with size from 2K to hundreds of MiBs.
///
/// Each cache entry will be aligned to a multiplier of 4K on disk, hence too small cache entries will lead to heavy
/// internal fragmentation.
///
/// The disk cache evicts cache entries in block unit.
pub struct BlockEngineConfig<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    device: Arc<dyn Device>,
    block_size: usize,
    compression: Compression,
    indexer_shards: usize,
    recover_concurrency: usize,
    flushers: usize,
    reclaimers: usize,
    buffer_pool_size: usize,
    blob_index_size: usize,
    submit_queue_size_threshold: usize,
    clean_block_threshold: usize,
    eviction_pickers: Vec<Box<dyn EvictionPicker>>,
    admission_filter: StorageFilter,
    reinsertion_filter: StorageFilter,
    enable_tombstone_log: bool,
    #[cfg(any(test, feature = "test_utils"))]
    flush_switch: Switch,
    #[cfg(any(test, feature = "test_utils"))]
    load_holder: Holder,
    entry_observer: Option<Arc<dyn EntryObserver>>,
    marker: PhantomData<(K, V, P)>,
}

impl<K, V, P> Debug for BlockEngineConfig<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockEngineConfig")
            .field("device", &self.device)
            .field("block_size", &self.block_size)
            .field("compression", &self.compression)
            .field("indexer_shards", &self.indexer_shards)
            .field("recover_concurrency", &self.recover_concurrency)
            .field("flushers", &self.flushers)
            .field("reclaimers", &self.reclaimers)
            .field("buffer_pool_size", &self.buffer_pool_size)
            .field("blob_index_size", &self.blob_index_size)
            .field("submit_queue_size_threshold", &self.submit_queue_size_threshold)
            .field("clean_block_threshold", &self.clean_block_threshold)
            .field("eviction_pickers", &self.eviction_pickers)
            .field("admission_filter", &self.admission_filter)
            .field("reinsertion_filter", &self.reinsertion_filter)
            .field("enable_tombstone_log", &self.enable_tombstone_log)
            .field("entry_observer", &self.entry_observer)
            .finish()
    }
}

impl<K, V, P> BlockEngineConfig<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    /// Create a new block-based disk cache engine builder with default configurations.
    pub fn new(device: Arc<dyn Device>) -> Self {
        Self {
            device,
            block_size: 16 * 1024 * 1024, // 16 MiB
            compression: Compression::default(),
            indexer_shards: 64,
            recover_concurrency: 8,
            flushers: 1,
            reclaimers: 1,
            buffer_pool_size: 16 * 1024 * 1024,            // 16 MiB
            blob_index_size: 4 * 1024,                     // 4 KiB
            submit_queue_size_threshold: 16 * 1024 * 1024, // 16 MiB
            clean_block_threshold: 1,
            eviction_pickers: vec![Box::new(InvalidRatioPicker::new(0.8)), Box::<FifoPicker>::default()],
            admission_filter: StorageFilter::new(),
            reinsertion_filter: StorageFilter::new().with_condition(RejectAll),
            enable_tombstone_log: false,
            #[cfg(any(test, feature = "test_utils"))]
            flush_switch: Switch::default(),
            #[cfg(any(test, feature = "test_utils"))]
            load_holder: Holder::default(),
            entry_observer: None,
            marker: PhantomData,
        }
    }

    /// Set the block size for the block-based disk cache engine.
    ///
    /// Block is the minimal cache eviction unit for the block-based disk cache,
    /// its size also limits the max cacheable entry size.
    ///
    /// The block size must be 4K-aligned. the given value is not 4K-aligned, it will be automatically aligned up.
    ///
    /// Default: `16 MiB`.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = bits::align_up(PAGE, block_size);
        self
    }

    /// Set the shard num of the indexer. Each shard has its own lock.
    ///
    /// Default: `64`.
    pub fn with_indexer_shards(mut self, indexer_shards: usize) -> Self {
        self.indexer_shards = indexer_shards;
        self
    }

    /// Set the recover concurrency for the disk cache store.
    ///
    /// Default: `8`.
    pub fn with_recover_concurrency(mut self, recover_concurrency: usize) -> Self {
        self.recover_concurrency = recover_concurrency;
        self
    }

    /// Set the flusher count for the disk cache store.
    ///
    /// The flusher count limits how many blocks can be concurrently written.
    ///
    /// Default: `1`.
    pub fn with_flushers(mut self, flushers: usize) -> Self {
        self.flushers = flushers;
        self
    }

    /// Set the admission filter for th disk cache store.
    ///
    /// The admission filter is used to pick the entries that can be inserted into the disk cache store.
    ///
    /// Default: Admit all.
    pub fn with_admission_filter(mut self, filter: StorageFilter) -> Self {
        self.admission_filter = filter;
        self
    }

    /// Set the reclaimer count for the disk cache store.
    ///
    /// The reclaimer count limits how many blocks can be concurrently reclaimed.
    ///
    /// Default: `1`.
    pub fn with_reclaimers(mut self, reclaimers: usize) -> Self {
        self.reclaimers = reclaimers;
        self
    }

    /// Set the total flush buffer pool size.
    ///
    /// Each flusher shares a volume at `threshold / flushers`.
    ///
    /// If the buffer of the flush queue exceeds the threshold, the further entries will be ignored.
    ///
    /// Default: 16 MiB.
    pub fn with_buffer_pool_size(mut self, buffer_pool_size: usize) -> Self {
        self.buffer_pool_size = buffer_pool_size;
        self
    }

    /// Set the blob index size for each blob.
    ///
    /// A larger blob index size can hold more blob entries, but it will also increase the io size of each blob part
    /// write.
    ///
    /// NOTE: The size will be aligned up to a multiplier of 4K.
    ///
    /// Default: 4 KiB
    pub fn with_blob_index_size(mut self, blob_index_size: usize) -> Self {
        let blob_index_size = bits::align_up(PAGE, blob_index_size);
        self.blob_index_size = blob_index_size;
        self
    }

    /// Set the submit queue size threshold.
    ///
    /// If the total entry estimated size in the submit queue exceeds the threshold, the further entries will be
    /// ignored.
    ///
    /// Default: `buffer_pool_size` * 2.
    pub fn with_submit_queue_size_threshold(mut self, submit_queue_size_threshold: usize) -> Self {
        self.submit_queue_size_threshold = submit_queue_size_threshold;
        self
    }

    /// Set the clean block threshold for the disk cache store.
    ///
    /// The reclaimers only work when the clean block count is equal to or lower than the clean block threshold.
    ///
    /// Default: the same value as the `reclaimers`.
    pub fn with_clean_block_threshold(mut self, clean_block_threshold: usize) -> Self {
        self.clean_block_threshold = clean_block_threshold;
        self
    }

    /// Set the eviction pickers for th disk cache store.
    ///
    /// The eviction picker is used to pick the block to reclaim.
    ///
    /// The eviction pickers are applied in order. If the previous eviction picker doesn't pick any block, the next one
    /// will be applied.
    ///
    /// If no eviction picker picks a block, a block will be picked randomly.
    ///
    /// Default: [ invalid ratio picker { threshold = 0.8 }, fifo picker ]
    pub fn with_eviction_pickers(mut self, eviction_pickers: Vec<Box<dyn EvictionPicker>>) -> Self {
        self.eviction_pickers = eviction_pickers;
        self
    }

    /// Set the reinsertion filter for th disk cache store.
    ///
    /// The reinsertion filter is used to pick the entries that can be reinsertion into the disk cache store while
    /// reclaiming.
    ///
    /// Note: Only extremely important entries should be picked. If too many entries are picked, both insertion and
    /// reinsertion will be stuck.
    ///
    /// Default: Reject all.
    pub fn with_reinsertion_filter(mut self, filter: StorageFilter) -> Self {
        self.reinsertion_filter = filter;
        self
    }

    /// Enable the tombstone log.
    ///
    /// For updatable cache, either the tombstone log or [`crate::engine::RecoverMode::None`] must be enabled to prevent
    /// from the phantom entries after reopen.
    pub fn with_tombstone_log(mut self, enable: bool) -> Self {
        self.enable_tombstone_log = enable;
        self
    }

    /// Set the observer of entry departures and recovery outcomes.
    ///
    /// Default: none.
    pub fn with_entry_observer(mut self, observer: Arc<dyn EntryObserver>) -> Self {
        self.entry_observer = Some(observer);
        self
    }

    /// Pass the flush holder for test.
    #[cfg(any(test, feature = "test_utils"))]
    pub fn with_flush_switch(mut self, flush_switch: Switch) -> Self {
        self.flush_switch = flush_switch;
        self
    }

    /// Pass the load holder for test.
    #[cfg(any(test, feature = "test_utils"))]
    pub fn with_load_holder(mut self, load_holder: Holder) -> Self {
        self.load_holder = load_holder;
        self
    }

    /// Build the block-based disk cache engine with the given configurations.
    pub async fn build(
        self: Box<Self>,
        EngineBuildContext {
            io_engine,
            metrics,
            spawner: runtime,
            recover_mode,
        }: EngineBuildContext,
    ) -> Result<Arc<BlockEngine<K, V, P>>> {
        let device = self.device;
        let block_size = self.block_size;

        let mut tombstones = vec![];

        let tombstone_log = if self.enable_tombstone_log {
            // TODO(MrCroxx): The tombstone log support multiples partitions for multiple device support.
            let mut partitions = vec![];

            let max_entries = device.capacity() / PAGE;
            let pages = max_entries / TombstoneLog::SLOTS_PER_PAGE
                + if max_entries.is_multiple_of(TombstoneLog::SLOTS_PER_PAGE) {
                    0
                } else {
                    1
                };
            let partition = device.create_partition(pages * PAGE)?;
            partitions.push(partition);

            let tombstone_log = TombstoneLog::open(partitions, io_engine.clone(), &mut tombstones).await?;
            Some(tombstone_log)
        } else {
            None
        };

        let indexer = Indexer::new(self.indexer_shards, self.entry_observer.clone());
        let submit_queue = SubmitQueue::new(self.submit_queue_size_threshold);

        #[expect(clippy::type_complexity)]
        let (flushers, rxs): (Vec<Flusher<K, V, P>>, Vec<UnboundedReceiver<Submission<K, V, P>>>) = (0..self.flushers)
            .map(|id| Flusher::<K, V, P>::new(id, metrics.clone()))
            .unzip();

        let reclaimer = Reclaimer::new(
            indexer.clone(),
            flushers.clone(),
            Arc::new(self.reinsertion_filter),
            self.blob_index_size,
            device.statistics().clone(),
            runtime.clone(),
        );
        let reclaimer: Arc<dyn ReclaimerTrait> = Arc::new(reclaimer);

        let block_manager = BlockManager::open(
            device.clone(),
            io_engine,
            block_size,
            self.eviction_pickers,
            reclaimer,
            self.reclaimers,
            self.clean_block_threshold,
            metrics.clone(),
            runtime.clone(),
        )?;
        let blocks = block_manager.blocks();

        if self.flushers + self.clean_block_threshold > blocks / 2 {
            tracing::warn!(
                "[block engine]: block-based object disk cache stable blocks count is too small, flusher [{flushers}] + clean block threshold [{clean_block_threshold}] (default = reclaimers) is supposed to be much larger than the block count [{blocks}]",
                flushers = self.flushers,
                clean_block_threshold = self.clean_block_threshold,
            );
        }

        let sequence = AtomicSequence::default();

        RecoverRunner::run(
            self.recover_concurrency,
            recover_mode,
            self.blob_index_size,
            (0..blocks as BlockId).collect_vec(),
            &sequence,
            &indexer,
            &block_manager,
            &tombstones,
            runtime.clone(),
            metrics.clone(),
            self.entry_observer.as_ref(),
        )
        .await?;

        let io_buffer_size = self.buffer_pool_size / self.flushers;
        for (flusher, rx) in flushers.iter().zip(rxs) {
            flusher.run(
                rx,
                block_size,
                io_buffer_size,
                self.blob_index_size,
                self.compression,
                indexer.clone(),
                block_manager.clone(),
                tombstone_log.clone(),
                metrics.clone(),
                &runtime,
                #[cfg(any(test, feature = "test_utils"))]
                self.flush_switch.clone(),
            )?;
        }

        let admission_filter = self.admission_filter.with_condition(IoThrottle);

        let inner = BlockEngineInner {
            admission_filter,
            device,
            indexer,
            block_manager,
            flushers,
            submit_queue,
            sequence,
            _spawner: runtime,
            active: AtomicBool::new(true),
            metrics,
            #[cfg(any(test, feature = "test_utils"))]
            flush_switch: self.flush_switch,
            #[cfg(any(test, feature = "test_utils"))]
            load_holder: self.load_holder,
        };
        let inner = Arc::new(inner);
        let engine = BlockEngine { inner };
        let engine = Arc::new(engine);
        Ok(engine)
    }
}

impl<K, V, P> EngineConfig<K, V, P> for BlockEngineConfig<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn build(self: Box<Self>, ctx: EngineBuildContext) -> BoxFuture<'static, Result<Arc<dyn Engine<K, V, P>>>> {
        async move { self.build(ctx).await.map(|e| e as Arc<dyn Engine<K, V, P>>) }.boxed()
    }
}

impl<K, V, P> From<BlockEngineConfig<K, V, P>> for Box<dyn EngineConfig<K, V, P>>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn from(builder: BlockEngineConfig<K, V, P>) -> Self {
        builder.boxed()
    }
}

/// Block-based disk cache engine.
pub struct BlockEngine<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    inner: Arc<BlockEngineInner<K, V, P>>,
}

impl<K, V, P> Debug for BlockEngine<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockEngine").finish()
    }
}

struct BlockEngineInner<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    admission_filter: StorageFilter,

    device: Arc<dyn Device>,

    indexer: Indexer,
    block_manager: BlockManager,

    flushers: Vec<Flusher<K, V, P>>,

    submit_queue: Arc<SubmitQueue>,

    sequence: AtomicSequence,

    _spawner: Spawner,

    active: AtomicBool,

    metrics: Arc<Metrics>,

    #[cfg(any(test, feature = "test_utils"))]
    flush_switch: Switch,

    #[cfg(any(test, feature = "test_utils"))]
    load_holder: Holder,
}

impl<K, V, P> Clone for BlockEngine<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<K, V, P> BlockEngine<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn wait(&self) -> impl Future<Output = ()> + Send + 'static {
        let flushers = self.inner.flushers.clone();
        let block_manager = self.inner.block_manager.clone();
        async move {
            join_all(flushers.iter().map(|flusher| flusher.wait())).await;
            block_manager.wait_reclaim().await;
        }
    }

    fn close(&self) -> BoxFuture<'static, Result<()>> {
        let this = self.clone();
        async move {
            this.inner.active.store(false, Ordering::Relaxed);
            this.inner.submit_queue.close();
            this.wait().await;
            Ok(())
        }
        .boxed()
    }

    #[cfg_attr(feature = "tracing", trace(name = "foyer::storage::engine::block::engine::enqueue"))]
    fn enqueue(&self, piece: PieceRef<K, V, P>, estimated_size: usize) {
        if !self.admits(&piece) {
            return;
        }
        match self.inner.submit_queue.try_reserve(estimated_size) {
            Some(reservation) => self.submit(piece, reservation),
            None => {
                self.inner.metrics.storage_queue_channel_overflow.increase(1);
                self.inner
                    .indexer
                    .report_dropped_write(piece.hash(), WriteDropReason::QueueFull);
            }
        }
    }

    fn enqueue_paced(
        &self,
        piece: PieceRef<K, V, P>,
        estimated_size: usize,
    ) -> impl Future<Output = ()> + Send + 'static {
        let this = self.clone();
        async move {
            if !this.admits(&piece) {
                return;
            }
            match this.inner.submit_queue.reserve_paced(estimated_size).await {
                Some(reservation) => this.submit(piece, reservation),
                None => this
                    .inner
                    .indexer
                    .report_dropped_write(piece.hash(), WriteDropReason::Closed),
            }
        }
    }

    /// Whether a piece may be written: the engine is open and the piece is not young.
    fn admits(&self, piece: &PieceRef<K, V, P>) -> bool {
        if !self.inner.active.load(Ordering::Relaxed) {
            tracing::warn!("cannot enqueue new entry after closed");
            self.inner
                .indexer
                .report_dropped_write(piece.hash(), WriteDropReason::Closed);
            return false;
        }

        tracing::trace!(
            hash = piece.hash(),
            age = ?piece.properties().age().unwrap_or_default(),
            "[block engine]: enqueue"
        );
        match piece.properties().age().unwrap_or_default() {
            Age::Fresh | Age::Old => true,
            Age::Young => {
                // skip write block engine if the entry is still young
                self.inner.metrics.storage_block_engine_enqueue_skip.increase(1);
                false
            }
        }
    }

    /// Pushes a piece when the submit queue has room; returns `false`, reporting nothing, when it is past its threshold.
    fn try_enqueue(&self, piece: PieceRef<K, V, P>, estimated_size: usize) -> bool {
        if !self.admits(&piece) {
            return true;
        }
        match self.inner.submit_queue.try_reserve(estimated_size) {
            Some(reservation) => {
                self.submit(piece, reservation);
                true
            }
            None => {
                self.inner.metrics.storage_queue_channel_overflow.increase(1);
                false
            }
        }
    }

    fn submit(&self, piece: PieceRef<K, V, P>, reservation: Reservation) {
        self.inner.flushers[piece.hash() as usize % self.inner.flushers.len()].submit_entry(
            piece,
            reservation,
            &self.inner.sequence,
        );
    }

    fn load(&self, hash: u64) -> impl Future<Output = Result<Load<K, V, P>>> + Send + 'static {
        tracing::trace!(hash, "[block engine]: load");

        #[cfg(any(test, feature = "test_utils"))]
        let load_holer = self.inner.load_holder.wait();

        let indexer = self.inner.indexer.clone();
        let metrics = self.inner.metrics.clone();
        let block_manager = self.inner.block_manager.clone();

        let load = async move {
            #[cfg(any(test, feature = "test_utils"))]
            load_holer.await;

            let addr = match indexer.get(hash) {
                Some(addr) => addr,
                None => {
                    return Ok(Load::Miss);
                }
            };

            tracing::trace!(hash, ?addr, "[block engine]: load");

            let block = block_manager.block(addr.block);
            if block.partition().statistics().is_read_throttled() {
                return Ok(Load::Throttled);
            }

            let buf = IoSliceMut::new(bits::align_up(PAGE, addr.len as _));
            let (buf, res) = block.read(Box::new(buf), addr.offset as _).await;
            match res {
                Ok(_) => {}
                Err(e) => {
                    tracing::error!(hash, ?addr, ?e, "[block engine load]: load error");
                    return Err(e);
                }
            }

            let header = match EntryHeader::read(&buf[..EntryHeader::serialized_len()]) {
                Ok(header) => header,
                Err(e) => {
                    return match e.kind() {
                        ErrorKind::Parse
                        | ErrorKind::MagicMismatch
                        | ErrorKind::ChecksumMismatch
                        | ErrorKind::OutOfRange => {
                            tracing::warn!(
                                hash,
                                ?addr,
                                ?e,
                                "[block engine load]: deserialize read buffer raise error, remove this entry and skip"
                            );
                            indexer.remove(hash, &addr, DepartureReason::ReadCorruption);
                            Ok(Load::Miss)
                        }
                        _ => {
                            tracing::error!(hash, ?addr, ?e, "[block engine load]: load error");
                            Err(e)
                        }
                    };
                }
            };

            let (key, value) = {
                let now = Instant::now();
                let res = match EntryDeserializer::deserialize::<K, V>(
                    &buf[EntryHeader::serialized_len()..],
                    header.key_len as _,
                    header.value_len as _,
                    header.compression,
                    Some(header.checksum),
                ) {
                    Ok(res) => res,
                    Err(e) => {
                        return match e.kind() {
                            ErrorKind::MagicMismatch | ErrorKind::ChecksumMismatch | ErrorKind::OutOfRange => {
                                tracing::warn!(
                                    hash,
                                    ?addr,
                                    ?header,
                                    ?e,
                                    "[block engine load]: deserialize read buffer raise error, remove this entry and skip"
                                );
                                indexer.remove(hash, &addr, DepartureReason::ReadCorruption);
                                Ok(Load::Miss)
                            }
                            _ => {
                                tracing::error!(hash, ?addr, ?header, ?e, "[block engine load]: load error");
                                Err(e)
                            }
                        };
                    }
                };
                metrics
                    .storage_entry_deserialize_duration
                    .record(now.elapsed().as_secs_f64());
                res
            };

            let age = match block.statistics().probation.load(Ordering::Relaxed) {
                true => Age::Old,
                false => Age::Young,
            };

            Ok(Load::Entry {
                key,
                value,
                populated: Populated { age },
            })
        };
        #[cfg(feature = "tracing")]
        let load = load.in_span(Span::enter_with_local_parent(
            "foyer::storage::engine::block::engine::load",
        ));
        load
    }

    fn delete(&self, hash: u64) {
        if !self.inner.active.load(Ordering::Relaxed) {
            tracing::warn!("cannot delete entry after closed");
            return;
        }

        let sequence = self.inner.sequence.fetch_add(1, Ordering::Relaxed);
        let stats = self
            .inner
            .indexer
            .insert_tombstone(hash, sequence)
            .map(|addr| InvalidStats {
                block: addr.block,
                size: bits::align_up(PAGE, addr.len as usize),
            });

        let this = self.clone();

        this.inner.flushers[hash as usize % this.inner.flushers.len()].submit(Submission::Tombstone {
            tombstone: Tombstone { hash, sequence },
            stats,
        });
    }

    fn may_contains(&self, hash: u64) -> bool {
        self.inner.indexer.get(hash).is_some()
    }

    fn destroy(&self) -> BoxFuture<'static, Result<()>> {
        let this = self.clone();
        async move {
            if !this.inner.active.load(Ordering::Relaxed) {
                return Err(Error::new(ErrorKind::Closed, "cannot delete entry after closed"));
            }

            // Write a tombstone to clear tombstone log by increase the max sequence.
            let sequence = this.inner.sequence.fetch_add(1, Ordering::Relaxed);

            this.inner.flushers[0].submit(Submission::Tombstone {
                tombstone: Tombstone { hash: 0, sequence },
                stats: None,
            });
            this.wait().await;

            // Clear indices.
            //
            // This step must perform after the latest writer finished,
            // otherwise the indices of the latest batch cannot be cleared.
            this.inner.indexer.clear();

            // Clean blocks.
            try_join_all((0..this.inner.block_manager.blocks() as BlockId).map(|id| {
                let block = this.inner.block_manager.block(id).clone();
                async move {
                    let res = BlockCleaner::clean(&block).await;
                    block.statistics().reset();
                    res
                }
            }))
            .await?;

            Ok(())
        }
        .boxed()
    }

    #[cfg(any(test, feature = "test_utils"))]
    pub fn hold_flush(&self) {
        self.inner.flush_switch.on();
    }

    #[cfg(any(test, feature = "test_utils"))]
    pub fn unhold_flush(&self) {
        self.inner.flush_switch.off();
    }
}

impl<K, V, P> Engine<K, V, P> for BlockEngine<K, V, P>
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    fn device(&self) -> &Arc<dyn Device> {
        &self.inner.device
    }

    fn filter(&self, hash: u64, estimated_size: usize) -> StorageFilterResult {
        self.inner
            .admission_filter
            .filter(self.inner.device.statistics(), hash, estimated_size)
    }

    fn enqueue(&self, piece: PieceRef<K, V, P>, estimated_size: usize) {
        self.enqueue(piece, estimated_size);
    }

    fn enqueue_paced(&self, piece: PieceRef<K, V, P>, estimated_size: usize) -> BoxFuture<'static, ()> {
        self.enqueue_paced(piece, estimated_size).boxed()
    }

    fn try_enqueue(&self, piece: PieceRef<K, V, P>, estimated_size: usize) -> bool {
        self.try_enqueue(piece, estimated_size)
    }

    fn load(&self, hash: u64) -> BoxFuture<'static, Result<Load<K, V, P>>> {
        // TODO(MrCroxx): refactor this.
        self.load(hash).boxed()
    }

    fn delete(&self, hash: u64) {
        self.delete(hash);
    }

    fn may_contains(&self, hash: u64) -> bool {
        self.may_contains(hash)
    }

    fn destroy(&self) -> BoxFuture<'static, Result<()>> {
        self.destroy()
    }

    fn entry_payload_bytes(&self) -> usize {
        self.inner.indexer.payload_bytes()
    }

    fn allocated_bytes(&self) -> usize {
        self.inner.block_manager.allocated_bytes()
    }

    fn wait(&self) -> BoxFuture<'static, ()> {
        // TODO(MrCroxx): refactor this.
        self.wait().boxed()
    }

    fn close(&self) -> BoxFuture<'static, Result<()>> {
        self.close()
    }
}

#[cfg(test)]
mod tests {

    use std::{fs::File, path::Path};

    use bytesize::ByteSize;
    use foyer_common::hasher::ModHasher;
    use foyer_memory::{Cache, CacheBuilder, CacheEntry, FifoConfig, TestProperties};
    use itertools::Itertools;

    use super::*;
    use crate::{
        PsyncIoEngineConfig, RejectAll,
        engine::{
            RecoverMode,
            block::observer::{Departure, DroppedWrite, RecoveryReport},
        },
        io::{
            device::{DeviceBuilder, combined::CombinedDeviceBuilder, file::FileDeviceBuilder, fs::FsDeviceBuilder},
            engine::{IoEngine, IoEngineBuildContext, IoEngineConfig},
        },
        serde::EntrySerializer,
        test_utils::Biased,
    };

    const KB: usize = 1024;

    fn cache_for_test() -> Cache<u64, Vec<u8>, ModHasher, TestProperties> {
        CacheBuilder::new(10)
            .with_shards(1)
            .with_eviction_config(FifoConfig::default())
            .with_hash_builder(ModHasher::default())
            .build()
    }

    async fn io_engine_for_test(spawner: Spawner) -> Arc<dyn IoEngine> {
        // TODO(MrCroxx): Test with other io engines.
        PsyncIoEngineConfig::new()
            .boxed()
            .build(IoEngineBuildContext { spawner })
            .await
            .unwrap()
    }

    /// 4 files, fifo eviction, 16 KiB block, 64 KiB capacity.
    async fn engine_for_test(dir: impl AsRef<Path>) -> Arc<BlockEngine<u64, Vec<u8>, TestProperties>> {
        store_for_test_with_reinsertion_filter(dir, StorageFilter::new().with_condition(RejectAll)).await
    }

    async fn store_for_test_with_reinsertion_filter(
        dir: impl AsRef<Path>,
        reinsertion_filter: StorageFilter,
    ) -> Arc<BlockEngine<u64, Vec<u8>, TestProperties>> {
        let device = FsDeviceBuilder::new(dir)
            .with_capacity(ByteSize::kib(64).as_u64() as _)
            .build()
            .unwrap();
        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;
        let metrics = Arc::new(Metrics::noop());
        let builder = BlockEngineConfig {
            device,
            block_size: 16 * 1024,
            compression: Compression::None,
            indexer_shards: 4,
            recover_concurrency: 2,
            flushers: 1,
            reclaimers: 1,
            clean_block_threshold: 1,
            admission_filter: StorageFilter::new(),
            eviction_pickers: vec![Box::<FifoPicker>::default()],
            reinsertion_filter,
            enable_tombstone_log: false,
            entry_observer: None,
            buffer_pool_size: 16 * 1024 * 1024,
            blob_index_size: 4 * 1024,
            submit_queue_size_threshold: 16 * 1024 * 1024 * 2,
            flush_switch: Switch::default(),
            load_holder: Holder::default(),
            marker: PhantomData,
        };

        let builder = Box::new(builder);
        builder
            .build(EngineBuildContext {
                io_engine,
                metrics,
                spawner,
                recover_mode: RecoverMode::Strict,
            })
            .await
            .unwrap()
    }

    async fn store_for_test_with_tombstone_log(
        dir: impl AsRef<Path>,
    ) -> Arc<BlockEngine<u64, Vec<u8>, TestProperties>> {
        let device = FsDeviceBuilder::new(dir)
            .with_capacity(ByteSize::kib(64).as_u64() as usize + ByteSize::kib(4).as_u64() as usize)
            .build()
            .unwrap();
        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;
        let metrics = Arc::new(Metrics::noop());
        let builder = BlockEngineConfig {
            device,
            block_size: 16 * 1024,
            compression: Compression::None,
            indexer_shards: 4,
            recover_concurrency: 2,
            flushers: 1,
            reclaimers: 1,
            clean_block_threshold: 1,
            eviction_pickers: vec![Box::<FifoPicker>::default()],
            admission_filter: StorageFilter::new(),
            reinsertion_filter: StorageFilter::new().with_condition(RejectAll),
            enable_tombstone_log: true,
            entry_observer: None,
            buffer_pool_size: 16 * 1024 * 1024,
            blob_index_size: 4 * 1024,
            submit_queue_size_threshold: 16 * 1024 * 1024 * 2,
            flush_switch: Switch::default(),
            load_holder: Holder::default(),
            marker: PhantomData,
        };
        let builder = Box::new(builder);
        builder
            .build(EngineBuildContext {
                io_engine,
                metrics,
                spawner,
                recover_mode: RecoverMode::Strict,
            })
            .await
            .unwrap()
    }

    fn enqueue(
        store: &BlockEngine<u64, Vec<u8>, TestProperties>,
        entry: CacheEntry<u64, Vec<u8>, ModHasher, TestProperties>,
    ) {
        let estimated_size = EntrySerializer::estimated_size(entry.key(), entry.value());
        store.enqueue(entry.piece().into(), estimated_size);
    }

    #[test_log::test(tokio::test)]
    async fn test_store_enqueue_lookup_recovery() {
        let dir = tempfile::tempdir().unwrap();

        let memory = cache_for_test();
        let store = engine_for_test(dir.path()).await;

        // [ [e1, e2], [], [], [] ]
        store.hold_flush();
        let e1 = memory.insert(1, vec![1; 7 * KB]);
        let e2 = memory.insert(2, vec![2; 3 * KB]);
        enqueue(&store, e1.clone());
        enqueue(&store, e2);
        store.unhold_flush();
        store.wait().await;

        let r1 = store.load(memory.hash(&1)).await.unwrap().kv().unwrap();
        assert_eq!(r1, (1, vec![1; 7 * KB]));
        let r2 = store.load(memory.hash(&2)).await.unwrap().kv().unwrap();
        assert_eq!(r2, (2, vec![2; 3 * KB]));

        // [ [e1, e2], [e3, e4], [], [] ]
        store.hold_flush();
        let e3 = memory.insert(3, vec![3; 7 * KB]);
        let e4 = memory.insert(4, vec![4; 2 * KB]);
        enqueue(&store, e3);
        enqueue(&store, e4);
        store.unhold_flush();
        store.wait().await;

        let r1 = store.load(memory.hash(&1)).await.unwrap().kv().unwrap();
        assert_eq!(r1, (1, vec![1; 7 * KB]));
        let r2 = store.load(memory.hash(&2)).await.unwrap().kv().unwrap();
        assert_eq!(r2, (2, vec![2; 3 * KB]));
        let r3 = store.load(memory.hash(&3)).await.unwrap().kv().unwrap();
        assert_eq!(r3, (3, vec![3; 7 * KB]));
        let r4 = store.load(memory.hash(&4)).await.unwrap().kv().unwrap();
        assert_eq!(r4, (4, vec![4; 2 * KB]));

        // [ [e1, e2], [e3, e4], [e5], [] ]
        let e5 = memory.insert(5, vec![5; 11 * KB]);
        enqueue(&store, e5);
        store.wait().await;

        let r1 = store.load(memory.hash(&1)).await.unwrap().kv().unwrap();
        assert_eq!(r1, (1, vec![1; 7 * KB]));
        let r2 = store.load(memory.hash(&2)).await.unwrap().kv().unwrap();
        assert_eq!(r2, (2, vec![2; 3 * KB]));
        let r3 = store.load(memory.hash(&3)).await.unwrap().kv().unwrap();
        assert_eq!(r3, (3, vec![3; 7 * KB]));
        let r4 = store.load(memory.hash(&4)).await.unwrap().kv().unwrap();
        assert_eq!(r4, (4, vec![4; 2 * KB]));
        let r5 = store.load(memory.hash(&5)).await.unwrap().kv().unwrap();
        assert_eq!(r5, (5, vec![5; 11 * KB]));

        // [ [], [e3, e4], [e5], [e6, e4*] ]
        store.hold_flush();
        let e6 = memory.insert(6, vec![6; 7 * KB]);
        let e4v2 = memory.insert(4, vec![!4; 3 * KB]);
        enqueue(&store, e6);
        enqueue(&store, e4v2);
        store.unhold_flush();
        store.wait().await;

        assert!(store.load(memory.hash(&1)).await.unwrap().kv().is_none());
        assert!(store.load(memory.hash(&2)).await.unwrap().kv().is_none());
        let r3 = store.load(memory.hash(&3)).await.unwrap().kv().unwrap();
        assert_eq!(r3, (3, vec![3; 7 * KB]));
        let r4v2 = store.load(memory.hash(&4)).await.unwrap().kv().unwrap();
        assert_eq!(r4v2, (4, vec![!4; 3 * KB]));
        let r5 = store.load(memory.hash(&5)).await.unwrap().kv().unwrap();
        assert_eq!(r5, (5, vec![5; 11 * KB]));
        let r6 = store.load(memory.hash(&6)).await.unwrap().kv().unwrap();
        assert_eq!(r6, (6, vec![6; 7 * KB]));

        store.close().await.unwrap();
        enqueue(&store, e1);
        store.wait().await;

        drop(store);

        let store = engine_for_test(dir.path()).await;

        assert!(store.load(memory.hash(&1)).await.unwrap().kv().is_none());
        assert!(store.load(memory.hash(&2)).await.unwrap().kv().is_none());
        let r3 = store.load(memory.hash(&3)).await.unwrap().kv().unwrap();
        assert_eq!(r3, (3, vec![3; 7 * KB]));
        let r4v2 = store.load(memory.hash(&4)).await.unwrap().kv().unwrap();
        assert_eq!(r4v2, (4, vec![!4; 3 * KB]));
        let r5 = store.load(memory.hash(&5)).await.unwrap().kv().unwrap();
        assert_eq!(r5, (5, vec![5; 11 * KB]));
        let r6 = store.load(memory.hash(&6)).await.unwrap().kv().unwrap();
        assert_eq!(r6, (6, vec![6; 7 * KB]));
    }

    #[test_log::test(tokio::test)]
    async fn test_store_delete_recovery() {
        let dir = tempfile::tempdir().unwrap();

        let memory = cache_for_test();
        let store = store_for_test_with_tombstone_log(dir.path()).await;

        let es = (0..10).map(|i| memory.insert(i, vec![i as u8; 3 * KB])).collect_vec();

        // [[0, 1, 2], [3, 4, 5], [6, 7, 8], []]
        for e in es.iter().take(9) {
            enqueue(&store, e.clone());
        }
        store.wait().await;

        for i in 0..9 {
            assert_eq!(
                store.load(memory.hash(&i)).await.unwrap().kv(),
                Some((i, vec![i as u8; 3 * KB]))
            );
        }

        store.delete(memory.hash(&3));
        store.wait().await;
        assert_eq!(store.load(memory.hash(&3)).await.unwrap().kv(), None);

        store.close().await.unwrap();
        drop(store);

        let store = store_for_test_with_tombstone_log(dir.path()).await;
        for i in 0..9 {
            if i != 3 {
                assert_eq!(
                    store.load(memory.hash(&i)).await.unwrap().kv(),
                    Some((i, vec![i as u8; 3 * KB]))
                );
            } else {
                assert_eq!(store.load(memory.hash(&3)).await.unwrap().kv(), None);
            }
        }

        enqueue(&store, es[3].clone());
        store.wait().await;
        assert_eq!(
            store.load(memory.hash(&3)).await.unwrap().kv(),
            Some((3, vec![3; 3 * KB]))
        );

        store.close().await.unwrap();
        drop(store);

        let store = store_for_test_with_tombstone_log(dir.path()).await;

        assert_eq!(
            store.load(memory.hash(&3)).await.unwrap().kv(),
            Some((3, vec![3; 3 * KB]))
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_store_destroy_recovery() {
        let dir = tempfile::tempdir().unwrap();

        let memory = cache_for_test();
        let store = store_for_test_with_tombstone_log(dir.path()).await;

        let es = (0..10).map(|i| memory.insert(i, vec![i as u8; 3 * KB])).collect_vec();

        // [[0, 1, 2], [3, 4, 5], [6, 7, 8], []]
        store.hold_flush();
        for e in es.iter().take(9) {
            enqueue(&store, e.clone());
        }
        store.unhold_flush();
        store.wait().await;

        for i in 0..9 {
            assert_eq!(
                store.load(memory.hash(&i)).await.unwrap().kv(),
                Some((i, vec![i as u8; 3 * KB]))
            );
        }

        store.delete(memory.hash(&3));
        store.wait().await;
        assert_eq!(store.load(memory.hash(&3)).await.unwrap().kv(), None);

        store.destroy().await.unwrap();

        store.close().await.unwrap();
        drop(store);

        let store = store_for_test_with_tombstone_log(dir.path()).await;
        for i in 0..9 {
            assert_eq!(store.load(memory.hash(&i)).await.unwrap().kv(), None);
        }

        enqueue(&store, es[3].clone());
        store.wait().await;
        assert_eq!(
            store.load(memory.hash(&3)).await.unwrap().kv(),
            Some((3, vec![3; 3 * KB]))
        );

        store.close().await.unwrap();
        drop(store);

        let store = store_for_test_with_tombstone_log(dir.path()).await;

        assert_eq!(
            store.load(memory.hash(&3)).await.unwrap().kv(),
            Some((3, vec![3; 3 * KB]))
        );
    }

    // FIXME(MrCroxx): Move the admission test to store level.
    // #[test_log::test(tokio::test)]
    // async fn test_store_admission() {
    //     let dir = tempfile::tempdir().unwrap();

    //     let memory = cache_for_test();
    //     let store = store_for_test_with_admission_picker(&memory, dir.path(),
    // Arc::new(BiasedPicker::new([1]))).await;

    //     let e1 = memory.insert(1, vec![1; 7 * KB]);
    //     let e2 = memory.insert(2, vec![2; 7 * KB]);

    //     assert!(enqueue(&store, e1.clone(),).await.unwrap());
    //     assert!(!enqueue(&store, e2,).await.unwrap());

    //     let r1 = store.load(&1).await.unwrap().unwrap();
    //     assert_eq!(r1, (1, vec![1; 7 * KB]));
    //     assert!(store.load(&2).await.unwrap().is_none());
    // }

    #[test_log::test(tokio::test)]
    async fn test_store_reinsertion() {
        let dir = tempfile::tempdir().unwrap();

        let memory = cache_for_test();
        let store = store_for_test_with_reinsertion_filter(
            dir.path(),
            StorageFilter::new().with_condition(Biased::new(vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19])),
        )
        .await;

        let es = (0..15).map(|i| memory.insert(i, vec![i as u8; 3 * KB])).collect_vec();

        // [[(0), (1), (2)], [(3), (4), (5)], [(6), (7), (8)], []]
        for e in es.iter().take(9).cloned() {
            enqueue(&store, e);
            store.wait().await;
        }

        for i in 0..9 {
            let r = store.load(memory.hash(&i)).await.unwrap().kv().unwrap();
            assert_eq!(r, (i, vec![i as u8; 3 * KB]));
        }

        // [[], [(3), (4), (5)], [(6), (7), (8)], [(9), (10), (1)]]
        enqueue(&store, es[9].clone());
        enqueue(&store, es[10].clone());
        store.wait().await;
        let mut res = vec![];
        for i in 0..11 {
            res.push(store.load(memory.hash(&i)).await.unwrap().kv());
        }
        assert_eq!(
            res,
            vec![
                None,
                Some((1, vec![1; 3 * KB])),
                None,
                Some((3, vec![3; 3 * KB])),
                Some((4, vec![4; 3 * KB])),
                Some((5, vec![5; 3 * KB])),
                Some((6, vec![6; 3 * KB])),
                Some((7, vec![7; 3 * KB])),
                Some((8, vec![8; 3 * KB])),
                Some((9, vec![9; 3 * KB])),
                Some((10, vec![10; 3 * KB])),
            ]
        );

        // [[(11), (3), (5)], [], [(6), (7), (8)], [(9), (10), (1)]]
        enqueue(&store, es[11].clone());
        store.wait().await;
        let mut res = vec![];
        for i in 0..12 {
            res.push(store.load(memory.hash(&i)).await.unwrap().kv());
        }
        assert_eq!(
            res,
            vec![
                None,
                Some((1, vec![1; 3 * KB])),
                None,
                Some((3, vec![3; 3 * KB])),
                None,
                Some((5, vec![5; 3 * KB])),
                Some((6, vec![6; 3 * KB])),
                Some((7, vec![7; 3 * KB])),
                Some((8, vec![8; 3 * KB])),
                Some((9, vec![9; 3 * KB])),
                Some((10, vec![10; 3 * KB])),
                Some((11, vec![11; 3 * KB])),
            ]
        );

        // [[(11), (3), (5)], [(12), (13), (14)], [], [(9), (10), (1)]]
        store.delete(memory.hash(&7));
        store.wait().await;
        enqueue(&store, es[12].clone());
        store.wait().await;
        enqueue(&store, es[13].clone());
        store.wait().await;
        enqueue(&store, es[14].clone());
        store.wait().await;
        let mut res = vec![];
        for i in 0..15 {
            res.push(store.load(memory.hash(&i)).await.unwrap().kv());
        }
        assert_eq!(
            res,
            vec![
                None,
                Some((1, vec![1; 3 * KB])),
                None,
                Some((3, vec![3; 3 * KB])),
                None,
                Some((5, vec![5; 3 * KB])),
                None,
                None,
                None,
                Some((9, vec![9; 3 * KB])),
                Some((10, vec![10; 3 * KB])),
                Some((11, vec![11; 3 * KB])),
                Some((12, vec![12; 3 * KB])),
                Some((13, vec![13; 3 * KB])),
                Some((14, vec![14; 3 * KB])),
            ]
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_store_magic_checksum_mismatch() {
        let dir = tempfile::tempdir().unwrap();

        let memory = cache_for_test();
        let store = engine_for_test(dir.path()).await;

        // write entry 1
        let e1 = memory.insert(1, vec![1; 7 * KB]);
        enqueue(&store, e1);
        store.wait().await;

        // check entry 1
        let r1 = store.load(memory.hash(&1)).await.unwrap().kv().unwrap();
        assert_eq!(r1, (1, vec![1; 7 * KB]));

        // corrupt entry and header
        for entry in std::fs::read_dir(dir.path()).unwrap() {
            let entry = entry.unwrap();
            if !entry.metadata().unwrap().is_file() {
                continue;
            }

            let file = File::options().write(true).open(entry.path()).unwrap();
            #[cfg(target_family = "unix")]
            {
                use std::os::unix::fs::FileExt;
                file.write_all_at(&[b'x'; 42], 5 * 1024).unwrap();
            }
            #[cfg(target_family = "windows")]
            {
                use std::os::windows::fs::FileExt;
                file.seek_write(&[b'x'; 42], 5 * 1024).unwrap();
            }
        }

        assert!(store.load(memory.hash(&1)).await.unwrap().kv().is_none());
    }

    #[test_log::test(tokio::test)]
    async fn test_aggregated_device() {
        let dir = tempfile::tempdir().unwrap();

        const KB: usize = 1024;
        const MB: usize = 1024 * 1024;

        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;

        let d1 = FsDeviceBuilder::new(dir.path().join("dev1"))
            .with_capacity(MB)
            .build()
            .unwrap();
        let d2 = FsDeviceBuilder::new(dir.path().join("dev2"))
            .with_capacity(2 * MB)
            .build()
            .unwrap();
        let d3 = FsDeviceBuilder::new(dir.path().join("dev3"))
            .with_capacity(4 * MB)
            .build()
            .unwrap();
        let device = CombinedDeviceBuilder::new()
            .with_device(d1)
            .with_device(d2)
            .with_device(d3)
            .build()
            .unwrap();
        let engine = BlockEngineConfig::<u64, Vec<u8>, TestProperties>::new(device)
            .with_block_size(64 * KB)
            .boxed()
            .build(EngineBuildContext {
                io_engine,
                metrics: Arc::new(Metrics::noop()),
                spawner,
                recover_mode: RecoverMode::None,
            })
            .await
            .unwrap();
        assert_eq!(engine.inner.block_manager.blocks(), (1 + 2 + 4) * MB / (64 * KB));
    }

    #[derive(Debug, Default)]
    struct Recorder {
        departures: parking_lot::Mutex<Vec<Departure>>,
        recoveries: parking_lot::Mutex<Vec<RecoveryReport>>,
        dropped: parking_lot::Mutex<Vec<DroppedWrite>>,
    }

    impl EntryObserver for Recorder {
        fn on_departure(&self, departure: Departure) {
            self.departures.lock().push(departure);
        }

        fn on_recovery(&self, report: RecoveryReport) {
            self.recoveries.lock().push(report);
        }

        fn on_dropped_write(&self, dropped: DroppedWrite) {
            self.dropped.lock().push(dropped);
        }
    }

    impl Recorder {
        fn take(&self) -> Vec<(u64, DepartureReason)> {
            std::mem::take(&mut *self.departures.lock())
                .into_iter()
                .map(|departure| (departure.hash, departure.reason))
                .collect()
        }
    }

    /// Same shape as [`store_for_test_with_reinsertion_filter`], observed.
    async fn engine_with_observer(
        dir: impl AsRef<Path>,
        observer: Arc<Recorder>,
        enable_tombstone_log: bool,
        reinsertion_filter: StorageFilter,
    ) -> Arc<BlockEngine<u64, Vec<u8>, TestProperties>> {
        let tombstone = if enable_tombstone_log { 4 * KB } else { 0 };
        let device = FsDeviceBuilder::new(dir)
            .with_capacity(64 * KB + tombstone)
            .build()
            .unwrap();
        engine_on_device(device, observer, enable_tombstone_log, reinsertion_filter).await
    }

    async fn engine_on_device(
        device: Arc<dyn Device>,
        observer: Arc<Recorder>,
        enable_tombstone_log: bool,
        reinsertion_filter: StorageFilter,
    ) -> Arc<BlockEngine<u64, Vec<u8>, TestProperties>> {
        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;
        let builder = BlockEngineConfig {
            device,
            block_size: 16 * KB,
            compression: Compression::None,
            indexer_shards: 4,
            recover_concurrency: 2,
            flushers: 1,
            reclaimers: 1,
            clean_block_threshold: 1,
            admission_filter: StorageFilter::new(),
            eviction_pickers: vec![Box::<FifoPicker>::default()],
            reinsertion_filter,
            enable_tombstone_log,
            entry_observer: Some(observer),
            buffer_pool_size: 16 * 1024 * 1024,
            blob_index_size: 4 * KB,
            submit_queue_size_threshold: 16 * 1024 * 1024 * 2,
            flush_switch: Switch::default(),
            load_holder: Holder::default(),
            marker: PhantomData,
        };
        Box::new(builder)
            .build(EngineBuildContext {
                io_engine,
                metrics: Arc::new(Metrics::noop()),
                spawner,
                recover_mode: RecoverMode::Quiet,
            })
            .await
            .unwrap()
    }

    /// Holds one flush buffer of a 16 KiB block and a 16 KiB submit queue: a few 3 KiB entries fill either.
    async fn engine_with_small_write_path(
        dir: &Path,
        observer: Arc<Recorder>,
    ) -> BlockEngine<u64, Vec<u8>, TestProperties> {
        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;
        let config = BlockEngineConfig {
            device: FsDeviceBuilder::new(dir).with_capacity(256 * KB).build().unwrap(),
            block_size: 16 * KB,
            compression: Compression::None,
            indexer_shards: 4,
            recover_concurrency: 2,
            flushers: 1,
            reclaimers: 1,
            clean_block_threshold: 1,
            admission_filter: StorageFilter::new(),
            eviction_pickers: vec![Box::<FifoPicker>::default()],
            reinsertion_filter: StorageFilter::new().with_condition(RejectAll),
            enable_tombstone_log: false,
            entry_observer: Some(observer),
            buffer_pool_size: 16 * KB,
            blob_index_size: 4 * KB,
            submit_queue_size_threshold: 16 * KB,
            flush_switch: Switch::default(),
            load_holder: Holder::default(),
            marker: PhantomData,
        };
        let engine = Box::new(config)
            .build(EngineBuildContext {
                io_engine,
                metrics: Arc::new(Metrics::noop()),
                spawner,
                recover_mode: RecoverMode::None,
            })
            .await
            .unwrap();
        (*engine).clone()
    }

    #[test_log::test(tokio::test)]
    async fn test_paced_writes_wait_for_the_flusher_and_unpaced_overflow_is_reported() {
        const ENTRIES: u64 = 12;
        let bound = std::time::Duration::from_secs(10);
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let recorder = Arc::new(Recorder::default());
        let store = engine_with_small_write_path(dir.path(), recorder.clone()).await;
        let piece = |key: u64| {
            let entry = memory.insert(key, vec![key as u8; 3 * KB]);
            let estimated_size = EntrySerializer::estimated_size(entry.key(), entry.value());
            (PieceRef::from(entry.piece()), estimated_size)
        };

        // Paced: with the flush held, the writer outruns the one buffer and waits; released, every entry lands.
        store.hold_flush();
        let pieces = (0..ENTRIES).map(piece).collect_vec();
        let writer = tokio::spawn({
            let store = store.clone();
            async move {
                for (piece, estimated_size) in pieces {
                    store.enqueue_paced(piece, estimated_size).await;
                }
            }
        });
        store.unhold_flush();
        let flushed = store.wait();
        tokio::time::timeout(bound, writer).await.unwrap().unwrap();
        tokio::time::timeout(bound, flushed).await.unwrap();
        tokio::time::timeout(bound, store.wait()).await.unwrap();
        for key in 0..ENTRIES {
            assert!(
                store.inner.indexer.get(memory.hash(&key)).is_some(),
                "entry {key} written"
            );
        }
        assert!(recorder.dropped.lock().is_empty());

        // Unpaced: past the submit queue threshold the write is dropped and reported.
        store.hold_flush();
        for key in ENTRIES..2 * ENTRIES {
            let (piece, estimated_size) = piece(key);
            store.enqueue(piece, estimated_size);
        }
        store.unhold_flush();
        tokio::time::timeout(bound, store.wait()).await.unwrap();
        let dropped = std::mem::take(&mut *recorder.dropped.lock());
        assert!(!dropped.is_empty());
        assert!(
            dropped
                .iter()
                .all(|dropped| dropped.reason == WriteDropReason::QueueFull)
        );
        let written = (ENTRIES..2 * ENTRIES)
            .filter(|key| store.inner.indexer.get(memory.hash(key)).is_some())
            .count();
        assert_eq!(written + dropped.len(), ENTRIES as usize);

        // An entry larger than the flush buffer is dropped as oversized; after close, writes are dropped as closed.
        let entry = memory.insert(u64::MAX, vec![0; 17 * KB]);
        let estimated_size = EntrySerializer::estimated_size(entry.key(), entry.value());
        store.enqueue_paced(entry.piece().into(), estimated_size).await;
        tokio::time::timeout(bound, store.wait()).await.unwrap();
        store.close().await.unwrap();
        let (piece, estimated_size) = piece(0);
        store.enqueue_paced(piece, estimated_size).await;
        let reasons = recorder
            .dropped
            .lock()
            .iter()
            .map(|dropped| dropped.reason)
            .collect_vec();
        assert_eq!(reasons, vec![WriteDropReason::Oversized, WriteDropReason::Closed]);
    }

    fn assert_payload_matches_index(store: &BlockEngine<u64, Vec<u8>, TestProperties>) {
        assert_eq!(store.entry_payload_bytes(), store.inner.indexer.indexed_payload_bytes());
    }

    fn payload(store: &BlockEngine<u64, Vec<u8>, TestProperties>, hash: u64) -> usize {
        store.inner.indexer.get(hash).unwrap().payload()
    }

    /// Flips one byte inside the only on-disk copy of `needle`.
    fn corrupt(dir: &Path, needle: &[u8]) {
        let mut found = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if !path.is_file() {
                continue;
            }
            let mut bytes = std::fs::read(&path).unwrap();
            if let Some(position) = bytes.windows(needle.len()).position(|window| window == needle) {
                bytes[position + needle.len() / 2] ^= 0x40;
                std::fs::write(&path, bytes).unwrap();
                found += 1;
            }
        }
        assert_eq!(found, 1, "exactly one persisted copy");
    }

    #[test_log::test(tokio::test)]
    async fn test_payload_gauge_follows_entry_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let recorder = Arc::new(Recorder::default());
        let store = engine_with_observer(
            dir.path(),
            recorder.clone(),
            false,
            StorageFilter::new().with_condition(RejectAll),
        )
        .await;
        let hash = |key: u64| memory.hash(&key);

        // Enqueued entries count only once flushed; an entry larger than a block is never accepted.
        store.hold_flush();
        enqueue(&store, memory.insert(1, vec![1; 7 * KB]));
        enqueue(&store, memory.insert(2, vec![2; 3 * KB]));
        enqueue(&store, memory.insert(3, vec![3; 17 * KB]));
        assert_eq!(store.entry_payload_bytes(), 0);
        store.unhold_flush();
        store.wait().await;
        let p1 = payload(&store, hash(1));
        let p2 = payload(&store, hash(2));
        assert!(
            (7 * KB..7 * KB + 64).contains(&p1),
            "payload {p1} excludes header and padding"
        );
        assert!(store.inner.indexer.get(hash(3)).is_none());
        assert_eq!(store.entry_payload_bytes(), p1 + p2);
        assert_payload_matches_index(&store);
        assert!(recorder.take().is_empty());

        // Replacement.
        let e1v2 = vec![!1; 2 * KB];
        enqueue(&store, memory.insert(1, e1v2.clone()));
        store.wait().await;
        let p1v2 = payload(&store, hash(1));
        assert_eq!(store.entry_payload_bytes(), p1v2 + p2);
        assert_eq!(recorder.take(), vec![(hash(1), DepartureReason::Replacement)]);

        // Explicit delete, repeated.
        store.delete(hash(2));
        store.delete(hash(2));
        store.wait().await;
        assert_eq!(store.entry_payload_bytes(), p1v2);
        assert_eq!(recorder.take(), vec![(hash(2), DepartureReason::ExplicitDelete)]);

        // Read-time corruption.
        corrupt(dir.path(), &e1v2);
        assert!(store.load(hash(1)).await.unwrap().kv().is_none());
        assert_eq!(store.entry_payload_bytes(), 0);
        assert_eq!(recorder.take(), vec![(hash(1), DepartureReason::ReadCorruption)]);

        // Destroy.
        enqueue(&store, memory.insert(4, vec![4; 3 * KB]));
        store.wait().await;
        store.destroy().await.unwrap();
        assert_eq!(store.entry_payload_bytes(), 0);
        assert_eq!(recorder.take(), vec![(hash(4), DepartureReason::ExplicitDelete)]);
    }

    #[cfg(target_family = "unix")]
    fn filesystem_allocated(dir: &Path) -> usize {
        use std::os::unix::fs::MetadataExt;
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().metadata().unwrap())
            .filter(|metadata| metadata.is_file())
            .map(|metadata| metadata.blocks() as usize * 512)
            .sum()
    }

    /// Deallocates every file under `dir` while keeping its length.
    #[cfg(target_family = "unix")]
    fn deallocate(dir: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
                let len = file.metadata().unwrap().len();
                file.set_len(0).unwrap();
                file.set_len(len).unwrap();
                file.sync_all().unwrap();
            }
        }
    }

    #[cfg(target_family = "unix")]
    #[test_log::test(tokio::test)]
    async fn test_allocated_bytes_count_writes_and_read_filesystem_at_open() {
        allocated_bytes_count_writes_and_read_filesystem_at_open(false).await;
    }

    /// Partitions of one file are ranges of it, so each block reads its own extent.
    #[cfg(target_family = "unix")]
    #[test_log::test(tokio::test)]
    async fn test_allocated_bytes_count_writes_and_read_filesystem_at_open_on_one_file() {
        allocated_bytes_count_writes_and_read_filesystem_at_open(true).await;
    }

    /// The running count comes from the engine's own writes, so reading it does no filesystem IO; the filesystem's
    /// allocation is read once, when the engine opens.
    #[cfg(target_family = "unix")]
    async fn allocated_bytes_count_writes_and_read_filesystem_at_open(one_file: bool) {
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let open = || async {
            let device = if one_file {
                FileDeviceBuilder::new(dir.path().join("blocks"))
                    .with_capacity(64 * KB)
                    .build()
                    .unwrap()
            } else {
                FsDeviceBuilder::new(dir.path()).with_capacity(64 * KB).build().unwrap()
            };
            engine_on_device(
                device,
                Arc::new(Recorder::default()),
                false,
                StorageFilter::new().with_condition(RejectAll),
            )
            .await
        };
        let store = open().await;
        assert_eq!(store.allocated_bytes(), 0);
        assert_eq!(filesystem_allocated(dir.path()), 0);

        enqueue(&store, memory.insert(1, vec![1; 7 * KB]));
        enqueue(&store, memory.insert(2, vec![2; 3 * KB]));
        store.wait().await;
        let allocated = store.allocated_bytes();
        assert!(allocated >= store.entry_payload_bytes() + 2 * EntryHeader::serialized_len());
        // A filesystem may allocate more than was written (APFS materializes a sparse file on its first write).
        assert!(allocated <= filesystem_allocated(dir.path()));

        // Replacing an entry leaves the obsolete record allocated.
        enqueue(&store, memory.insert(1, vec![!1; 2 * KB]));
        store.wait().await;
        let replaced = store.allocated_bytes();
        assert!(replaced > allocated);
        assert!(replaced <= filesystem_allocated(dir.path()));
        store.close().await.unwrap();
        drop(store);

        let store = open().await;
        let reopened = store.allocated_bytes();
        assert_eq!(reopened, filesystem_allocated(dir.path()));
        assert!(reopened >= replaced);

        // Reading the count does not look at the filesystem again.
        deallocate(dir.path());
        assert_eq!(filesystem_allocated(dir.path()), 0);
        assert_eq!(store.allocated_bytes(), reopened);
        store.close().await.unwrap();
        drop(store);

        let store = open().await;
        assert_eq!(store.allocated_bytes(), 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_payload_gauge_follows_reclaim_and_relocation() {
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let recorder = Arc::new(Recorder::default());
        let store = engine_with_observer(
            dir.path(),
            recorder.clone(),
            false,
            StorageFilter::new().with_condition(Biased::new(vec![memory.hash(&1)])),
        )
        .await;
        let es = (0..11).map(|i| memory.insert(i, vec![i as u8; 3 * KB])).collect_vec();

        // [[0, 1, 2], [3, 4, 5], [6, 7, 8], []]
        for e in es.iter().take(9).cloned() {
            enqueue(&store, e);
            store.wait().await;
        }
        let before = store.entry_payload_bytes();
        let p0 = payload(&store, memory.hash(&0));
        let p2 = payload(&store, memory.hash(&2));

        // Reclaiming the first block drops 0 and 2 and relocates 1: [[], [3, 4, 5], [6, 7, 8], [9, 10, 1]]
        enqueue(&store, es[9].clone());
        enqueue(&store, es[10].clone());
        store.wait().await;
        let mut departures = recorder.take();
        departures.sort();
        let mut expected = vec![
            (memory.hash(&0), DepartureReason::Reclaim),
            (memory.hash(&2), DepartureReason::Reclaim),
        ];
        expected.sort();
        assert_eq!(departures, expected);
        assert!(store.load(memory.hash(&1)).await.unwrap().kv().is_some());
        assert_eq!(
            store.entry_payload_bytes(),
            before - p0 - p2 + payload(&store, memory.hash(&9)) + payload(&store, memory.hash(&10))
        );
        assert_payload_matches_index(&store);
    }

    #[test_log::test(tokio::test)]
    async fn test_recovery_restores_only_live_entries() {
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let recorder = Arc::new(Recorder::default());
        let store = engine_with_observer(
            dir.path(),
            recorder.clone(),
            true,
            StorageFilter::new().with_condition(RejectAll),
        )
        .await;
        assert_eq!(*recorder.recoveries.lock(), vec![RecoveryReport::default()]);

        for (key, value) in [
            (1, vec![1; 3 * KB]),
            (1, vec![!1; 2 * KB]),
            (2, vec![2; 3 * KB]),
            (3, vec![3; 3 * KB]),
        ] {
            enqueue(&store, memory.insert(key, value));
            store.wait().await;
        }
        store.delete(memory.hash(&3));
        store.wait().await;
        let live = store.entry_payload_bytes();
        recorder.take();
        store.close().await.unwrap();
        drop(store);

        let recorder = Arc::new(Recorder::default());
        let store = engine_with_observer(
            dir.path(),
            recorder.clone(),
            true,
            StorageFilter::new().with_condition(RejectAll),
        )
        .await;
        assert_eq!(store.entry_payload_bytes(), live);
        assert_payload_matches_index(&store);
        assert_eq!(
            *recorder.recoveries.lock(),
            vec![RecoveryReport {
                restored_entries: 2,
                restored_payload_bytes: live,
                discarded_records: 2,
                truncated_records: 0,
                corrupt_blocks: 0,
            }]
        );
        let mut departures = recorder.take();
        departures.sort();
        let mut expected = vec![
            (memory.hash(&1), DepartureReason::RecoveryDiscard),
            (memory.hash(&3), DepartureReason::RecoveryDiscard),
        ];
        expected.sort();
        assert_eq!(departures, expected);
    }

    /// A 16 MiB device of 1 MiB blocks with a submit queue no test burst fills: nothing is reclaimed or dropped.
    async fn engine_for_many_entries(dir: &Path, observer: Arc<Recorder>) -> BlockEngine<u64, Vec<u8>, TestProperties> {
        let spawner = Spawner::current();
        let io_engine = io_engine_for_test(spawner.clone()).await;
        let config = BlockEngineConfig {
            device: FsDeviceBuilder::new(dir).with_capacity(16 * 1024 * KB).build().unwrap(),
            block_size: 1024 * KB,
            compression: Compression::None,
            indexer_shards: 4,
            recover_concurrency: 2,
            flushers: 1,
            reclaimers: 1,
            clean_block_threshold: 1,
            admission_filter: StorageFilter::new(),
            eviction_pickers: vec![Box::<FifoPicker>::default()],
            reinsertion_filter: StorageFilter::new().with_condition(RejectAll),
            enable_tombstone_log: false,
            entry_observer: Some(observer),
            buffer_pool_size: 1024 * KB,
            blob_index_size: 4 * KB,
            submit_queue_size_threshold: 64 * 1024 * KB,
            flush_switch: Switch::default(),
            load_holder: Holder::default(),
            marker: PhantomData,
        };
        let engine = Box::new(config)
            .build(EngineBuildContext {
                io_engine,
                metrics: Arc::new(Metrics::noop()),
                spawner,
                recover_mode: RecoverMode::Quiet,
            })
            .await
            .unwrap();
        (*engine).clone()
    }

    /// Threads racing to enqueue into one flusher: recovery restores every entry they enqueued and finds none behind
    /// a lower sequence.
    #[test_log::test(tokio::test)]
    async fn test_concurrent_submitters_recover_every_entry() {
        const SUBMITTERS: u64 = 8;
        const ENTRIES: u64 = 128;
        for round in 0..16 {
            let dir = tempfile::tempdir().unwrap();
            let memory = cache_for_test();
            let recorder = Arc::new(Recorder::default());
            let store = engine_for_many_entries(dir.path(), recorder.clone()).await;
            std::thread::scope(|scope| {
                for submitter in 0..SUBMITTERS {
                    let (store, memory) = (&store, &memory);
                    scope.spawn(move || {
                        for key in submitter * ENTRIES..(submitter + 1) * ENTRIES {
                            let entry = memory.insert(key, vec![key as u8; 64]);
                            let estimated_size = EntrySerializer::estimated_size(entry.key(), entry.value());
                            store.enqueue(entry.piece().into(), estimated_size);
                        }
                    });
                }
            });
            store.close().await.unwrap();
            assert!(recorder.dropped.lock().is_empty(), "round {round}");
            drop(store);

            let recorder = Arc::new(Recorder::default());
            let store = engine_for_many_entries(dir.path(), recorder.clone()).await;
            let report = recorder.recoveries.lock()[0];
            assert_eq!(report.truncated_records, 0, "round {round}");
            assert_eq!(report.restored_entries, (SUBMITTERS * ENTRIES) as usize, "round {round}");
            for key in 0..SUBMITTERS * ENTRIES {
                assert!(
                    store.inner.indexer.get(memory.hash(&key)).is_some(),
                    "round {round}: entry {key}"
                );
            }
        }
    }

    /// Records written behind a lower sequence in one block are counted, not restored.
    #[test_log::test(tokio::test)]
    async fn test_recovery_counts_records_behind_a_lower_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let memory = cache_for_test();
        let store = engine_for_many_entries(dir.path(), Arc::new(Recorder::default())).await;
        store.hold_flush();
        for (key, sequence) in [(1u64, 10), (2, 11), (3, 5), (4, 12)] {
            let entry = memory.insert(key, vec![key as u8; 64]);
            let estimated_size = EntrySerializer::estimated_size(entry.key(), entry.value());
            let reservation = store.inner.submit_queue.try_reserve(estimated_size).unwrap();
            store.inner.flushers[0].submit(Submission::CacheEntry {
                piece: entry.piece().into(),
                reservation,
                sequence,
            });
        }
        store.unhold_flush();
        store.close().await.unwrap();
        drop(store);

        let recorder = Arc::new(Recorder::default());
        let store = engine_for_many_entries(dir.path(), recorder.clone()).await;
        let report = recorder.recoveries.lock()[0];
        assert_eq!((report.restored_entries, report.truncated_records), (2, 2));
        for (key, restored) in [(1, true), (2, true), (3, false), (4, false)] {
            assert_eq!(
                store.inner.indexer.get(memory.hash(&key)).is_some(),
                restored,
                "entry {key}"
            );
        }
    }
}

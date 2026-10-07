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
    collections::VecDeque,
    fmt::Debug,
    fs::File,
    mem::ManuallyDrop,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(feature = "tracing")]
use fastrace::prelude::*;
use foyer_common::{
    error::{Error, Result},
    spawn::Spawner,
};
use futures_core::future::BoxFuture;
use futures_util::FutureExt;

use crate::{
    RawFile,
    io::{
        bytes::{IoB, IoBuf, IoBufMut, Raw},
        device::Partition,
        engine::{IoEngine, IoEngineBuildContext, IoEngineConfig, IoHandle},
    },
};

#[derive(Debug)]
struct FileHandle(ManuallyDrop<File>);

#[cfg(target_family = "windows")]
impl From<RawFile> for FileHandle {
    fn from(raw: RawFile) -> Self {
        use std::os::windows::io::FromRawHandle;
        let file = unsafe { File::from_raw_handle(raw.0) };
        let file = ManuallyDrop::new(file);
        Self(file)
    }
}

#[cfg(target_family = "unix")]
impl From<RawFile> for FileHandle {
    fn from(raw: RawFile) -> Self {
        use std::os::unix::io::FromRawFd;
        let file = unsafe { File::from_raw_fd(raw.0) };
        let file = ManuallyDrop::new(file);
        Self(file)
    }
}

impl Deref for FileHandle {
    type Target = File;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for FileHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Config for synchronous I/O engine with pread(2)/pwrite(2).
#[derive(Debug)]
pub struct PsyncIoEngineConfig {
    read_threads: Option<NonZeroUsize>,

    #[cfg(any(test, feature = "test_utils"))]
    write_io_latency: Option<std::ops::Range<std::time::Duration>>,

    #[cfg(any(test, feature = "test_utils"))]
    read_io_latency: Option<std::ops::Range<std::time::Duration>>,
}

impl Default for PsyncIoEngineConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl From<PsyncIoEngineConfig> for Box<dyn IoEngineConfig> {
    fn from(builder: PsyncIoEngineConfig) -> Self {
        builder.boxed()
    }
}

impl PsyncIoEngineConfig {
    /// Create a new synchronous I/O engine config with default configurations.
    pub fn new() -> Self {
        Self {
            read_threads: None,
            #[cfg(any(test, feature = "test_utils"))]
            write_io_latency: None,
            #[cfg(any(test, feature = "test_utils"))]
            read_io_latency: None,
        }
    }

    /// Run reads on a pool of `threads` threads of their own instead of the spawner's blocking pool.
    ///
    /// The pool bounds how many reads reach the device at once; every read is queued as it is issued and
    /// completes on its own, so a caller with many reads in flight waits only for the device. Writes stay on
    /// the spawner's blocking pool, so a long write never holds a read thread.
    pub fn with_read_threads(mut self, threads: NonZeroUsize) -> Self {
        self.read_threads = Some(threads);
        self
    }

    /// Set the simulated additional write I/O latency for testing purposes.
    #[cfg(any(test, feature = "test_utils"))]
    pub fn with_write_io_latency(mut self, latency: std::ops::Range<std::time::Duration>) -> Self {
        self.write_io_latency = Some(latency);
        self
    }

    /// Set the simulated additional read I/O latency for testing purposes.
    #[cfg(any(test, feature = "test_utils"))]
    pub fn with_read_io_latency(mut self, latency: std::ops::Range<std::time::Duration>) -> Self {
        self.read_io_latency = Some(latency);
        self
    }
}

impl IoEngineConfig for PsyncIoEngineConfig {
    fn build(self: Box<Self>, ctx: IoEngineBuildContext) -> BoxFuture<'static, Result<Arc<dyn IoEngine>>> {
        async move {
            let reads = self.read_threads.map(ReadPool::start).transpose()?;
            let engine = PsyncIoEngine {
                spawner: ctx.spawner,
                reads,
                #[cfg(any(test, feature = "test_utils"))]
                write_io_latency: self.write_io_latency,
                #[cfg(any(test, feature = "test_utils"))]
                read_io_latency: self.read_io_latency,
            };
            let engine: Arc<dyn IoEngine> = Arc::new(engine);
            Ok(engine)
        }
        .boxed()
    }
}

/// The synchronous I/O engine that uses pread(2)/pwrite(2) and tokio thread pool for reading and writing.
pub struct PsyncIoEngine {
    spawner: Spawner,
    reads: Option<ReadPool>,

    #[cfg(any(test, feature = "test_utils"))]
    write_io_latency: Option<std::ops::Range<std::time::Duration>>,
    #[cfg(any(test, feature = "test_utils"))]
    read_io_latency: Option<std::ops::Range<std::time::Duration>>,
}

impl Debug for PsyncIoEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PsyncIoEngine").finish()
    }
}

impl IoEngine for PsyncIoEngine {
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::io::engine::psync::read")
    )]
    fn read(&self, buf: Box<dyn IoBufMut>, partition: &dyn Partition, offset: u64) -> IoHandle {
        let (raw, offset) = partition.translate(offset);
        let file = FileHandle::from(raw);
        let runtime = self.spawner.clone();

        #[cfg(feature = "tracing")]
        let span = Span::enter_with_local_parent("foyer::storage::io::engine::psync::read::io");

        #[cfg(any(test, feature = "test_utils"))]
        let read_io_latency = self.read_io_latency.clone();
        let read = move || {
            let (ptr, len) = buf.as_raw_parts();
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            let res = {
                #[cfg(target_family = "windows")]
                {
                    use std::os::windows::fs::FileExt;
                    file.seek_read(slice, offset).map(|_| ()).map_err(Error::io_error)
                }
                #[cfg(target_family = "unix")]
                {
                    use std::os::unix::fs::FileExt;
                    file.read_exact_at(slice, offset).map_err(Error::io_error)
                }
            };
            #[cfg(any(test, feature = "test_utils"))]
            if let Some(lat) = read_io_latency {
                std::thread::sleep(rand::random_range(lat));
            }
            (buf, res)
        };
        if let Some(reads) = &self.reads {
            let (tx, rx) = tokio::sync::oneshot::channel();
            reads.submit(Box::new(move || {
                // The reader may have stopped waiting; the buffer is dropped with the reply.
                let _ = tx.send(read());
            }));
            return async move {
                match rx.await {
                    Ok((buf, res)) => (buf.into_iob(), res),
                    Err(_) => (
                        Box::new(Raw::new(0)) as Box<dyn IoB>,
                        Err(Error::io_error(std::io::Error::other("the read pool stopped"))),
                    ),
                }
            }
            .boxed()
            .into();
        }
        async move {
            let (buf, res) = match runtime.spawn_blocking(read).await {
                Ok((buf, res)) => {
                    #[cfg(feature = "tracing")]
                    drop(span);
                    (buf, res)
                }
                Err(e) => return (Box::new(Raw::new(0)) as Box<dyn IoB>, Err(e)),
            };
            let buf: Box<dyn IoB> = buf.into_iob();
            (buf, res)
        }
        .boxed()
        .into()
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::io::engine::psync::write")
    )]
    fn write(&self, buf: Box<dyn IoBuf>, partition: &dyn Partition, offset: u64) -> IoHandle {
        let (raw, offset) = partition.translate(offset);
        let file = FileHandle::from(raw);
        let runtime = self.spawner.clone();

        #[cfg(feature = "tracing")]
        let span = Span::enter_with_local_parent("foyer::storage::io::engine::psync::write::io");

        #[cfg(any(test, feature = "test_utils"))]
        let write_io_latency = self.write_io_latency.clone();
        async move {
            let (buf, res) = match runtime
                .spawn_blocking(move || {
                    let (ptr, len) = buf.as_raw_parts();
                    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                    let res = {
                        #[cfg(target_family = "windows")]
                        {
                            use std::os::windows::fs::FileExt;
                            file.seek_write(slice, offset).map(|_| ()).map_err(Error::io_error)
                        }
                        #[cfg(target_family = "unix")]
                        {
                            use std::os::unix::fs::FileExt;
                            file.write_all_at(slice, offset).map_err(Error::io_error)
                        }
                    };
                    #[cfg(any(test, feature = "test_utils"))]
                    if let Some(lat) = write_io_latency {
                        std::thread::sleep(rand::random_range(lat));
                    }
                    (buf, res)
                })
                .await
            {
                Ok((buf, res)) => {
                    #[cfg(feature = "tracing")]
                    drop(span);
                    (buf, res)
                }
                Err(e) => return (Box::new(Raw::new(0)) as Box<dyn IoB>, Err(e)),
            };
            let buf: Box<dyn IoB> = buf.into_iob();
            (buf, res)
        }
        .boxed()
        .into()
    }
}

type ReadJob = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct ReadQueue {
    jobs: parking_lot::Mutex<(VecDeque<ReadJob>, bool)>,
    ready: parking_lot::Condvar,
    live_threads: AtomicUsize,
}

/// A fixed set of threads that run queued reads in order. Dropping it lets the threads finish the queued reads
/// and exit.
struct ReadPool {
    queue: Arc<ReadQueue>,
}

impl ReadPool {
    fn start(threads: NonZeroUsize) -> Result<Self> {
        let queue = Arc::new(ReadQueue::default());
        for _ in 0..threads.get() {
            let queue = Arc::clone(&queue);
            queue.live_threads.fetch_add(1, Ordering::Relaxed);
            std::thread::Builder::new()
                .name("foyer-read".into())
                .spawn(move || {
                    while let Some(job) = queue.next() {
                        job();
                    }
                    queue.live_threads.fetch_sub(1, Ordering::Relaxed);
                })
                .map_err(Error::io_error)?;
        }
        Ok(Self { queue })
    }

    fn submit(&self, job: ReadJob) {
        self.queue.jobs.lock().0.push_back(job);
        self.queue.ready.notify_one();
    }
}

impl ReadQueue {
    /// The next queued read; `None` once the pool is dropped and the queue is empty.
    fn next(&self) -> Option<ReadJob> {
        let mut jobs = self.jobs.lock();
        loop {
            if let Some(job) = jobs.0.pop_front() {
                return Some(job);
            }
            if jobs.1 {
                return None;
            }
            self.ready.wait(&mut jobs);
        }
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        self.queue.jobs.lock().1 = true;
        self.queue.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_util::future::join_all;
    use tempfile::tempdir;

    use super::*;
    use crate::io::{
        bytes::IoSliceMut,
        device::{Device, DeviceBuilder, file::FileDeviceBuilder},
    };

    const PAGE: usize = 4096;

    async fn pooled(
        threads: usize,
        latency: Option<Duration>,
    ) -> (Arc<dyn IoEngine>, Arc<dyn Device>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let device = FileDeviceBuilder::new(dir.path().join("device"))
            .with_capacity(4 * 1024 * 1024)
            .build()
            .unwrap();
        device.create_partition(1024 * 1024).unwrap();
        let mut config = PsyncIoEngineConfig::new().with_read_threads(NonZeroUsize::new(threads).unwrap());
        if let Some(latency) = latency {
            config = config.with_read_io_latency(latency..latency + Duration::from_nanos(1));
        }
        let engine = config
            .boxed()
            .build(IoEngineBuildContext {
                spawner: Spawner::current(),
            })
            .await
            .unwrap();
        let mut page = Box::new(IoSliceMut::new(64 * PAGE));
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = (i / PAGE) as u8;
        }
        let (_, res) = engine.write(page, device.partition(0).as_ref(), 0).await;
        res.unwrap();
        (engine, device, dir)
    }

    async fn read_page(engine: &Arc<dyn IoEngine>, device: &Arc<dyn Device>, page: usize) -> Result<u8> {
        let (buf, res) = engine
            .read(
                Box::new(IoSliceMut::new(PAGE)),
                device.partition(0).as_ref(),
                (page * PAGE) as u64,
            )
            .await;
        res.map(|()| {
            assert!(buf.iter().all(|byte| *byte == buf[0]));
            buf[0]
        })
    }

    #[test_log::test(tokio::test)]
    async fn a_failed_read_fails_alone_and_the_pool_keeps_serving() {
        let (engine, device, _dir) = pooled(2, None).await;
        let past_end = engine.read(
            Box::new(IoSliceMut::new(PAGE)),
            device.partition(0).as_ref(),
            (8 * 1024 * 1024) as u64,
        );
        let reads = (0..16).map(|page| read_page(&engine, &device, page));
        let (failed, pages) = tokio::join!(past_end, join_all(reads));
        assert!(failed.1.is_err());
        for (page, read) in pages.into_iter().enumerate() {
            assert_eq!(read.unwrap(), page as u8);
        }
        assert_eq!(read_page(&engine, &device, 63).await.unwrap(), 63);
    }

    #[test_log::test(tokio::test)]
    async fn dropped_reads_run_out_and_later_reads_complete() {
        let (engine, device, _dir) = pooled(2, Some(Duration::from_millis(1))).await;
        let abandoned = (0..64)
            .map(|page| {
                engine.read(
                    Box::new(IoSliceMut::new(PAGE)),
                    device.partition(0).as_ref(),
                    (page * PAGE) as u64,
                )
            })
            .collect::<Vec<_>>();
        drop(abandoned);
        assert_eq!(read_page(&engine, &device, 7).await.unwrap(), 7);
    }

    #[test_log::test(tokio::test)]
    async fn a_backlog_drains_at_most_the_pool_size_at_once() {
        const THREADS: usize = 4;
        let (engine, device, _dir) = pooled(THREADS, Some(Duration::from_millis(2))).await;
        let started = std::time::Instant::now();
        let reads = (0..64 * THREADS).map(|i| read_page(&engine, &device, i % 64));
        for (i, read) in join_all(reads).await.into_iter().enumerate() {
            assert_eq!(read.unwrap(), (i % 64) as u8);
        }
        // 256 reads of 2 ms on 4 threads take at least 64 x 2 ms; more threads would finish sooner.
        assert!(started.elapsed() >= Duration::from_millis(128));
    }

    #[test_log::test(tokio::test)]
    async fn the_pool_threads_exit_with_the_engine() {
        let pool = ReadPool::start(NonZeroUsize::new(3).unwrap()).unwrap();
        let queue = Arc::clone(&pool.queue);
        let (tx, rx) = tokio::sync::oneshot::channel();
        pool.submit(Box::new(move || tx.send(()).unwrap()));
        rx.await.unwrap();
        assert_eq!(queue.live_threads.load(Ordering::Relaxed), 3);
        let (exited_tx, exited_rx) = std::sync::mpsc::channel();
        let watched = Arc::clone(&queue);
        let watcher = std::thread::spawn(move || {
            while watched.live_threads.load(Ordering::Relaxed) != 0 {
                std::thread::yield_now();
            }
            exited_tx.send(()).unwrap();
        });
        drop(pool);
        exited_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        watcher.join().unwrap();
    }
}

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

use std::{collections::VecDeque, fmt::Debug, future::Future, sync::Arc};

use asyncband::oneshot;
use futures_util::future::{Either, ready};
use parking_lot::Mutex;

/// Bytes of cache entries submitted to the flushers and not yet taken into a flush buffer.
///
/// A paced writer waits here, in arrival order, until its entry fits under the threshold; an unpaced writer that finds
/// the queue past the threshold is refused.
#[derive(Debug)]
pub struct SubmitQueue {
    threshold: usize,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    bytes: usize,
    closed: bool,
    waiters: VecDeque<(usize, oneshot::Sender<Reservation>)>,
}

impl State {
    /// An entry larger than the threshold still enters an empty queue.
    fn fits(&self, bytes: usize, threshold: usize) -> bool {
        self.bytes == 0 || self.bytes + bytes <= threshold
    }
}

/// Queue bytes held by one submitted entry; dropping it returns them and admits waiting writers.
#[derive(Debug)]
pub struct Reservation {
    queue: Arc<SubmitQueue>,
    bytes: usize,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.bytes != 0 {
            self.queue.release(self.bytes);
        }
    }
}

impl SubmitQueue {
    pub fn new(threshold: usize) -> Arc<Self> {
        Arc::new(Self {
            threshold,
            state: Mutex::default(),
        })
    }

    fn reserve(self: &Arc<Self>, state: &mut State, bytes: usize) -> Reservation {
        state.bytes += bytes;
        Reservation {
            queue: self.clone(),
            bytes,
        }
    }

    /// Reserves `bytes` unless the queue is closed or already past its threshold.
    pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        let mut state = self.state.lock();
        (!state.closed && state.bytes <= self.threshold).then(|| self.reserve(&mut state, bytes))
    }

    /// Reserves `bytes` once they fit behind the writers already waiting; `None` if the queue closes first.
    pub fn reserve_paced(self: &Arc<Self>, bytes: usize) -> impl Future<Output = Option<Reservation>> + Send + 'static {
        let mut state = self.state.lock();
        if state.closed {
            return Either::Left(ready(None));
        }
        if state.waiters.is_empty() && state.fits(bytes, self.threshold) {
            return Either::Left(ready(Some(self.reserve(&mut state, bytes))));
        }
        let (tx, rx) = oneshot::channel();
        state.waiters.push_back((bytes, tx));
        Either::Right(async move { rx.await.ok() })
    }

    fn release(self: &Arc<Self>, bytes: usize) {
        let mut state = self.state.lock();
        state.bytes -= bytes;
        while let Some(&(bytes, _)) = state.waiters.front()
            && state.fits(bytes, self.threshold)
        {
            let (bytes, tx) = state.waiters.pop_front().unwrap();
            let reservation = self.reserve(&mut state, bytes);
            if let Err(error) = tx.send(reservation) {
                // The writer stopped waiting: undo the reservation under the held lock, emptied so its drop does not
                // lock again.
                let mut reservation = error.into_inner();
                state.bytes -= std::mem::take(&mut reservation.bytes);
            }
        }
    }

    /// Refuses every later reservation and wakes the waiting writers empty-handed.
    pub fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        state.waiters.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;

    use futures_util::FutureExt;

    use super::*;

    #[test_log::test(tokio::test)]
    async fn test_paced_writers_wait_in_order_and_unpaced_writers_are_refused_past_the_threshold() {
        let queue = SubmitQueue::new(10);
        let first = queue.reserve_paced(6).await.unwrap();
        let unpaced = queue.try_reserve(5).unwrap();
        assert!(queue.try_reserve(1).is_none(), "past the threshold");

        let mut second = pin!(queue.reserve_paced(4));
        let mut third = pin!(queue.reserve_paced(1));
        assert!(second.as_mut().now_or_never().is_none());
        assert!(third.as_mut().now_or_never().is_none());

        drop(unpaced);
        // 6 + 4 fits; the later small writer does not overtake it.
        let second = second.await.unwrap();
        assert!(third.as_mut().now_or_never().is_none());
        drop(first);
        let third = third.await.unwrap();
        drop((second, third));
        assert_eq!(queue.state.lock().bytes, 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_oversized_entry_enters_an_empty_queue_and_abandoned_waiters_return_their_bytes() {
        let queue = SubmitQueue::new(10);
        let oversized = queue.reserve_paced(64).await.unwrap();
        let abandoned = queue.reserve_paced(1);
        drop(abandoned);
        drop(oversized);
        assert_eq!(queue.state.lock().bytes, 0);

        let held = queue.reserve_paced(10).await.unwrap();
        let waiting = queue.reserve_paced(1);
        queue.close();
        assert!(waiting.await.is_none());
        assert!(queue.reserve_paced(1).await.is_none());
        assert!(queue.try_reserve(1).is_none());
        drop(held);
        assert_eq!(queue.state.lock().bytes, 0);
    }
}

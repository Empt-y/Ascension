//! Running the server on its own thread, and the client's end of the connection to it.
//!
//! The thread waits for client messages and handles each as it arrives, and ticks at a fixed
//! 20 Hz in between. A slow tick only slows the game's clock (TPS below 20), never the client's
//! frame rate. A network link would carry the same messages; in one process they go over channels
//! as they are.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};

use crate::protocol::{ClientMsg, ServerMsg};
use crate::server::{Server, TPS};

const TICK: Duration = Duration::from_nanos(1_000_000_000 / TPS as u64);
/// If the server falls this many ticks behind it gives up catching up (the clock slows instead).
const MAX_BEHIND: u32 = 5;

/// The client's end: send messages, receive the server's.
pub struct LocalLink {
    tx: Option<Sender<ClientMsg>>,
    rx: Receiver<ServerMsg>,
    thread: Option<JoinHandle<()>>,
}

impl LocalLink {
    pub fn send(&self, msg: ClientMsg) {
        if let Some(tx) = &self.tx {
            // A closed channel means the server thread has stopped (and logged why).
            let _ = tx.send(msg);
        }
    }

    pub fn try_recv(&self) -> Option<ServerMsg> {
        self.rx.try_recv().ok()
    }

    /// Wait up to `timeout` for the next message.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<ServerMsg> {
        self.rx.recv_timeout(timeout).ok()
    }
}

impl Drop for LocalLink {
    fn drop(&mut self) {
        // Closing our end stops the server loop.
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Start `server` on its own thread.
pub fn spawn(mut server: Server) -> LocalLink {
    let (client_tx, client_rx) = crossbeam_channel::unbounded::<ClientMsg>();
    let (server_tx, server_rx) = crossbeam_channel::unbounded::<ServerMsg>();
    let thread = std::thread::Builder::new()
        .name("server".into())
        .spawn(move || run(&mut server, &client_rx, &server_tx))
        .expect("spawning the server thread");
    LocalLink { tx: Some(client_tx), rx: server_rx, thread: Some(thread) }
}

fn run(server: &mut Server, rx: &Receiver<ClientMsg>, tx: &Sender<ServerMsg>) {
    let mut next = Instant::now() + TICK;
    let mut stats = Stats::default();
    loop {
        let now = Instant::now();
        if now >= next {
            let t = Instant::now();
            server.tick();
            stats.tick(t.elapsed());
            next += TICK;
            if now > next + TICK * MAX_BEHIND {
                next = now + TICK;
            }
            if let Some((tps, mspt)) = stats.report() {
                tx.send(ServerMsg::Stats { tps, mspt }).ok();
            }
        }
        for msg in server.drain() {
            if tx.send(msg).is_err() {
                return;
            }
        }
        match rx.recv_timeout(next.saturating_duration_since(Instant::now())) {
            Ok(msg) => {
                server.handle(msg);
                // Handle whatever else is already waiting before going back to the clock.
                while let Ok(msg) = rx.try_recv() {
                    server.handle(msg);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Ticks per second and the worst tick time, reported once a second.
#[derive(Default)]
struct Stats {
    since: Option<Instant>,
    ticks: u32,
    worst: Duration,
}

impl Stats {
    fn tick(&mut self, took: Duration) {
        self.since.get_or_insert_with(Instant::now);
        self.ticks += 1;
        self.worst = self.worst.max(took);
    }

    fn report(&mut self) -> Option<(f32, f32)> {
        let elapsed = self.since?.elapsed();
        if elapsed < Duration::from_secs(1) {
            return None;
        }
        let out = (self.ticks as f32 / elapsed.as_secs_f32(), self.worst.as_secs_f32() * 1e3);
        *self = Self::default();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::Config;

    #[test]
    fn the_server_thread_ticks_answers_in_order_and_stops_with_the_client() {
        let spec = asc_build::BuildingSpec::parse(include_str!("../../../assets/data/building.toml")).unwrap();
        let dir = std::env::temp_dir().join(format!("ascension-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = Config { save_dir: dir.clone(), seed: Some(5), new_world: false, check_reach: true, stress: None };
        let server = Server::open(config, asc_build::Lattice::new(&spec)).unwrap();
        let link = spawn(server);
        link.send(ClientMsg::SetTime(0.5));
        link.send(ClientMsg::Ping(1));
        // Everything sent before the ping is answered before the pong.
        let mut saw_time = false;
        let mut ticks = 0;
        let mut stats = None;
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(3) && (stats.is_none() || ticks < 20) {
            match link.recv_timeout(Duration::from_millis(100)) {
                Some(ServerMsg::Time { time_of_day, .. }) if (time_of_day - 0.5).abs() < 0.01 => {
                    saw_time = true;
                    ticks += 1;
                }
                Some(ServerMsg::Pong(1)) => assert!(saw_time, "pong before the reply to SetTime"),
                Some(ServerMsg::Stats { tps, mspt }) => stats = Some((tps, mspt)),
                _ => {}
            }
        }
        let (tps, mspt) = stats.expect("stats within a second or so");
        assert!(tps > 15.0 && tps < 25.0, "tps {tps}");
        assert!(mspt < 50.0, "mspt {mspt}");
        assert!(ticks >= 20);
        // Dropping the link joins the thread (this would hang if it didn't stop).
        drop(link);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

Hybrid Telemetry Bridge (low-latency + durable)
Goal

Keep SQLite + gRPC exactly as they are for durability and replay, but introduce a zero-copy, shared-memory fast lane for hot UI signals (frames, step counters, a few scalars). The GUI reads this fast lane directly; the daemon keeps persisting and broadcasting as today.

Architecture at a glance

Producer (your worker proxy or worker): writes frames + tiny HUD metrics into a shared-memory ring buffer (SPSC: single producer, single consumer).

Consumer (GUI): pulls from ring at the screen refresh cadence; when ring is empty, falls back to the existing gRPC stream.

Durability: unchanged—daemon still ingests via gRPC and batches into SQLite/WAL in the background.

Why this works: Intel Coach feels instantaneous because its dashboard tails files and the renderer runs in-process—no cross-process marshaling. We replicate the in-process feel with a shared-memory ring, while keeping your gRPC/SQLite pipeline for correctness and replay. Coach’s own dashboard is a Bokeh server that tails CSVs in the experiment directory to update charts live; it’s fast precisely because it reads locally appended data. 
Intel Labs

Ring buffer design (cross-platform, lock-free)

Memory layout (packed, cache-aligned):

struct Header {
  uint32  version;
  uint32  flags;             // bit 0: running, bit 1: paused …
  uint64  capacity;          // number of slots
  alignas(64) std::atomic<uint64_t> head; // producer writes
  alignas(64) std::atomic<uint64_t> tail; // consumer reads
  // hot scalars for HUD (rolling reward, fps, step/sec, etc.)
  double  last_reward;
  double  avg_reward_1s;
  double  step_rate_hz;
};

struct Slot {
  uint64  seq;               // seqlock generation to prevent torn reads
  uint32  w; uint32 h;       // frame dims
  uint32  fmt;               // e.g., RGB8
  uint32  bytes;             // payload length
  uint8   payload[bytes];    // tight-packed frame
};


SPSC flow: producer writes seq+1, copies payload, then writes seq+2 (even = stable). Consumer spins until seq is even and unchanged pre/post copy.

Per-run SHM name: mosaic.run.<run_id>.fastlane.

Cleanup: producer unlinks on process exit; GUI unlinks stale segments on run termination.

APIs you already have:

In Python workers/proxies: multiprocessing.shared_memory.SharedMemory gives you a named, POSIX-backed segment on Linux/macOS and a file mapping on Windows (one API, cross-platform). 
Python documentation

In the Qt/GUI side (if you prefer Qt types): QSharedMemory + QSystemSemaphore exist, but since your GUI is Python+Qt6, sticking to Python’s shared_memory is simplest and fastest. Qt’s class is there if/when you move this to C++. 
Qt Documentation

What goes through the ring (only the hot path):

Latest RGB frame (or tiled small textures if that’s cheaper).

A handful of scalars: episode index, step, reward, step/sec, loss* (if cheap).

Optional: compressed 8-bit heatmaps or value overlays instead of full frames when bandwidth is tight.

Everything else (per-step protobufs, long histograms, episodes, checkpoints) stays on gRPC → SQLite.

SQLite: keep it, tune it

Ensure WAL mode and batched transactions (you already use WAL). WAL reduces writer/reader contention and is ideal for concurrent read-most workloads. Use WAL checkpoints sensibly to avoid stalls. 
SQLite

For write latency, set PRAGMA synchronous=NORMAL (not FULL) on the telemetry DB; it meaningfully lowers fsync overhead while still being safe enough for this use-case. (See SQLite synchronous pragma docs for semantics.) 
SQLite

High-FPS Qt Rendering (ditch PyGame/Pyglet windows)

Your UI is Qt; use Qt’s native GPU pipeline for frames. Qt Quick’s Scene Graph renders via the Qt Rendering Hardware Interface (RHI) across Vulkan/Metal/D3D/OpenGL, with a dedicated render thread. This is the right tool for smooth, high-rate visuals inside a Qt app.

What to use (and avoid)

Prefer: QQuickWindow/QQuickView (QML/Qt Quick). This is the fast path that sits directly on the scene graph/render thread.

Avoid: QQuickWidget embedded in a QWidget hierarchy for high-FPS rendering—Qt explicitly calls out performance drawbacks; use QQuickWindow instead when you need throughput.

Two proven ways to feed frames to Qt Quick
1) QQuickFramebufferObject (FBO path)

Create a QQuickFramebufferObject (or its Python binding) and upload each frame to a texture in Renderer::render(). Qt handles the offscreen FBO and composes it into the scene graph on the render thread. This pattern is the canonical hook for custom real-time content in Qt Quick.

Why it’s good here

Runs on the render thread (no GUI stutters).

Easy to layer HUD/overlays in QML on top (text, charts, badges).

Works across RHI backends transparently in Qt 6.

2) Scene-graph texture node (zero-copy friendly)

Implement a QQuickItem that produces a QSGSimpleTextureNode fed by a QSGTexture you update each frame. If you later move to C++ for even lower overhead, you can use RHI directly (QRhi texture uploads, PBOs) and hook into QQuickWindow::beforeRendering to update the texture just-in-time.

(Note: Qt 6’s RHI abstracts the underlying API; when/if you go native, you’ll see classes like QRhiGraphicsPipeline, QRhiShaderStage, etc., but you do not need to drop to these unless you’re writing custom GPU code.) 
Qt Documentation

Frame scheduling & threading

Use a producer thread (or your SHM consumer thread) to map the latest ring slot into a CPU buffer and signal the QML item.

In QML/Qt Quick, call QQuickWindow::update(); Qt will render on its own render thread at vsync, pulling the latest texture.

If you need fixed-rate rendering (e.g., 60/120 Hz independent of UI), drive a precise timer on the Qt render side and simply sample the most recent ring slot.

Putting it together (step-by-step)
Phase A — Fast lane

Add fastlane writer in trainer_telemetry_proxy.py:

Create a named SharedMemory per run.

Allocate Header + N*Slot (N = 64–256).

On each new frame/step: write scalar HUD metrics into Header; copy the frame into the next Slot (seqlock), bump head.

Add fastlane reader in the GUI:

Open SharedMemory by name when a run becomes READY.

Start a lightweight consumer thread that polls head != tail, reads the latest stable slot, then sets tail = head (drop intermediate slots to stay real-time).

Backpressure integration (optional now, nicer later):

If GUI falls behind, show a counter; no need to pause training immediately.

Later, wire credits through your planned ControlStream: when dropped-frame ratio > threshold, send pause or lower a max_rate_hz. (Your proto already sketches this path.)

Phase B — Qt Quick rendering

New QML scene with one of:

QQuickFramebufferObject item that displays the latest frame texture.

Or a QQuickItem producing a QSGSimpleTextureNode fed from the shared buffer.

Overlays & HUD:

Render episode/step/reward with QML Text over the texture.

If you publish tiny 8-bit overlays (value/visitation maps) alongside the frame in the ring, draw them as separate semi-transparent textures.

UI knobs:

“Live FPS cap” (e.g., 30/60/120).

“Drop frames to stay real-time” (on by default).

“Prefer SHM fast lane / fall back to gRPC”.

Phase C — Keep persistence perfect

No change to your daemon’s PublishRun* handlers and SQLite sink.

Tune SQLite (if not already): WAL + synchronous=NORMAL + batch size (64/128) to reduce commit pressure. 
SQLite
+1

Why not PyGame/Pyglet/Godot here?

You’re already in Qt; embedding another window stack adds copies and focus/input headaches. Qt Quick’s scene graph runs on the GPU with its own render thread and is the first-class high-performance path inside Qt. Godot is great for games but heavyweight to embed; PyGame/Pyglet create separate GL contexts and fight the event loop. Qt itself recommends QQuickWindow/QQuickView over QQuickWidget for performance-sensitive views.

Validation & metrics

Target latency: time from worker step → pixels on screen.

SHM copy (CPU): 0.1–1.0 ms for ~84 KiB RGB (160×160) frames; ~2–5 ms for 84×84×3 with Python copies (opt: NumPy memoryview).

Texture upload: a few ms depending on GPU/driver; keep textures persistent and reuse them to avoid allocations.

End-to-end target: <16 ms (60 FPS) on mainstream hardware; faster if you skip full frames and render overlays.

Measure:

GUI: smoothed frame_time_ms, dropped frames, UI thread jank.

Daemon: SQLite batch commit time, WAL checkpoint pauses.

Transport: gRPC step rate vs. GUI frame rate (they can differ; GUI intentionally displays the latest, not every, step).

What to build first (minimal viable)

SHM ring (64 slots) + QQuickFramebufferObject viewer showing frames at 60 FPS.

HUD scalars in header; overlay with QML Text.

Drop-frames policy: always show the freshest slot; never block producer.

Keep SQLite untouched; just measure.

Optional: a toggle to bypass the JSONL→protobuf hop for local CleanRL runs (direct protobuf producer lowers one conversion).

This gets you Coach-like “live feel”, inside your Qt app, while your current multi-process, replayable architecture keeps doing its job.

References

Coach dashboard tails experiment CSVs to update charts: Intel Coach “Dashboard” docs. 
Intel Labs

Qt Quick / Scene Graph (the high-performance rendering path in Qt 6; RHI under the hood).

Avoid QQuickWidget for high FPS; prefer QQuickWindow/QQuickView.

QQuickFramebufferObject for custom real-time content.

Qt shared-memory primitives (QSharedMemory, QSystemSemaphore) if you later move the reader to C++. 
Qt Documentation

Python shared memory for the producer/GUI fast lane (cross-platform). 
Python documentation

SQLite WAL characteristics and write behavior; use WAL + batched transactions; consider synchronous=NORMAL. 
SQLite
+1

If you want, I can draft the SHM header/slot struct and a tiny PyQt6 QQuickFramebufferObject sample to drop into your repo next.




OTHER opinions


Absolutely—here’s a concrete, engineering-level plan to get Coach-like responsiveness without giving up MOSAIC’s distributed & durable telemetry.

Hybrid Telemetry Bridge (fast path + durable path)
1) Architecture at a glance

Keep your current gRPC → RunBus → SQLite pipeline for durability, replay, and multi-process workers.

Add a shared-memory ring buffer for hot visuals (frames + a tiny struct of “now” metrics). Notify the GUI with a local, event-driven channel. The GUI paints directly from shared memory using Qt Quick’s scene graph.

Worker/Proxy
  └─ writes latest RGBA/NV12 frame + metrics → shm ring (overwrite oldest)
  └─ signals "new head" via QLocalSocket/UDS
  └─ still sends step/episode summaries via gRPC (durable)

Trainer/GUI
  └─ listens on QLocalSocket -> reads ring head
  └─ wraps the shm pointer w/ QImage (no extra copy)
  └─ uploads to a persistent GPU texture (Qt Quick scene graph) and repaints
  └─ SQLite persists the slow path as today


Why this works:

Shared memory removes JSON/Protobuf (de)serialization and DB latency from visual updates.

You still get replay & crash-safety from SQLite + gRPC.

Overwrite semantics (ring) avoid backpressure stalls; visuals don’t need every frame.

2) IPC & memory layout (portable and simple)

Transport

Control/notify: QLocalSocket (Qt’s local IPC socket) for “frame N is ready” events; integrates with Qt’s event loop out of the box.

Data: QSharedMemory (or POSIX shm_open from the proxy, but the GUI side still attaches via Qt). It’s exactly for “fast data sharing between processes.”

Ring buffer header (in shared memory, 64-byte aligned)

struct ShmHeader {
  uint32 magic;          // 'MOSA'
  uint32 version;        // 1
  uint32 capacity;       // number of frame slots
  uint32 w, h;           // image size
  uint32 fourcc;         // e.g., BGRA / NV12
  uint32 stride;         // bytes per row for plane 0
  uint64 head;           // monotonically increasing frame counter
  uint64 tail;           // last frame the GUI has sampled (optional)
  // room for metrics: reward, eps, fps, step_idx...
  double last_reward;
  double rolling_return;
  double fps_estimate;
};


Frames

Slot size = plane sizes rounded up to 64-byte boundaries.

Formats:

BGRA8 (fastest for Qt upload; one plane).

Optional NV12 (half bandwidth; do simple shader in the Qt item to convert to RGB).

Write path (proxy/worker)

Next slot = head % capacity.

Write pixels into slot; update header fields.

head++; send tiny notify message on the local socket (e.g., 8-byte seq).

Read path (GUI)

On readyRead(), pull the latest head.

Compute slot; wrap the address as a QImage that references external memory (no copy).

Upload once into a persistent GPU texture and update() the Qt Quick item.

3) Qt-native rendering (no Pyglet/Pygame/Godot)

Pick one of these Qt Quick approaches; both use the scene graph and are built for high-FPS UIs:

Option A — QQuickFramebufferObject (FBO-based)

Implement a QQuickFramebufferObject::Renderer that, on each render(), updates a texture from the shm-backed QImage and draws a textured quad. Designed for custom rendering inside Qt Quick.

Option B — QQuickItem::updatePaintNode + QSGTexture

Subclass QQuickItem; in updatePaintNode(), keep a QSGSimpleTextureNode with a reused QSGTexture. When a new frame arrives, refresh the texture’s content and return the node. This is the canonical path for pushing dynamic images into the Qt Quick scene graph.

Performance guidance

Qt Quick’s scene graph runs a render thread; avoid copying on the GUI thread; only signal “new frame”.

Follow the Scene Graph performance tips: avoid per-frame allocs, reuse textures/nodes, keep paint nodes stable.

Advanced (later)

If you need deeper control over the backend (Vulkan/Metal/GL/D3D), Qt 6’s QRhi API underpins textures/buffers; you can create/update a QRhiTexture and wrap it as a scene-graph texture. This is the modern low-level path in Qt 6+. 
Qt Documentation

4) Zero-copy(ish) from shm to GPU

CPU: Use external-buffer QImage to avoid a CPU copy (ctor that wraps pointers). Upload to a persistent GPU texture each frame. (The GPU upload is unavoidable; minimize it by matching format BGRA8 to the window’s preferred format.)

GPU: Keep a single QSGTexture alive; just upload/setTexture with new pixels; don’t recreate textures per frame. (This follows scene-graph best practices. )

Notify via QLocalSocket → QMetaObject::invokeMethod(item, "newFrame") → update(); never block the GUI thread.

5) Backpressure & fairness (simple and robust)

Ring overwrites oldest if GUI lags—no stalls, visuals stay “present.”

(Optional) if head - tail exceeds a threshold, have the proxy downsample visual frames (e.g., every 2nd/4th) while keeping all step/episode records on the durable path.

You can add a per-run gRPC advisory rate or use your planned ControlStream tokens later; start simple first.

6) Where this plugs into your code (surgical edits)

Proxy side (trainer_telemetry_proxy.py)

Add SharedFramePublisher: allocates shm (name mosaic.run.<run_id>.frames), writes header, publishes frames when it sees {"type":"frame"} JSONL or when it can synthesize from envs (CleanRL: use capture-video hook).

Send notify bytes on a UNIX domain socket mosaic.run.<run_id>.sock each time head++.

GUI side

Add ShmFrameSource (C++/Python): attaches to shm, maps header + slots, opens QLocalSocket to ...sock, emits frameReady(seq).

Add QuickShmViewerItem (Qt Quick item): holds a persistent QSGTexture; on frameReady, wraps the slot with a QImage and updates the texture; repaints via updatePaintNode().

Wire into existing Live Tab; if shm is present, prefer it; else fall back to gRPC stream.

No change to your SQLite classes; they continue to persist steps/episodes for replay.

7) Tuning the slow path (today, no code changes)

SQLite: keep WAL, batch 64–128 records/commit for step/episode inserts.

GUI: throttle paint to vsync; add a “visual sampling” knob (e.g., draw every Nth frame) per run.

gRPC: coalesce scalar telemetry (not frames) to ~30–60 Hz if producers are chatty.

8) Expected gains

Visual frame p50 latency: ~~15–40 ms → 3–8 ms (notify + memcpy into GPU) on the same machine.

CPU relief on the GUI: remove JSON→proto→normalization on the hot path for frames.

No loss of replay/durability; SQLite remains authoritative for analysis.

9) Risks & mitigations

Lifetime of external buffers: when wrapping shm with QImage, ensure the shm segment outlives the paint; keep a ref-count or fence the upload before advancing head.

Cross-platform quirks: QSharedMemory works on Windows/macOS/Linux; stick to it on the GUI side to avoid platform #ifdefs.

GPU upload cost: unavoidable; minimize by matching formats and reusing textures (scene-graph perf guide).

10) Minimal milestone plan (1–2 iterations)

Milestone A (prototype, ~1–2 days of coding)

Proxy: dump one BGRA8 frame/second into shm + notify.

GUI: QuickShmViewerItem using updatePaintNode() + QSGSimpleTextureNode; renders the frame.

Toggle in UI: “Use fast visual path (shared memory)”.

Milestone B (productionize)

Ring with overwrite, per-run socket, metrics in header.

Opt-in for CleanRL/BDI workers.

Scene-graph cleanups per performance doc (persistent nodes, no allocs in hot loop).

11) Why Qt over Pyglet/Pygame/Godot for this

You’re already on Qt; Qt Quick Scene Graph is purpose-built for high-FPS UIs, runs a render thread, and integrates with the rest of your app’s models & signals—no second windowing stack.

QQuickFramebufferObject / updatePaintNode are first-class APIs for custom textured rendering.

QLocalSocket/QSharedMemory give you in-process-like latency while staying cross-platform and event-driven.

TL;DR

Yes: add a hybrid telemetry bridge—shared-memory ring + local notify for visuals; keep SQLite/gRPC for everything else.

Yes: use Qt Quick’s scene graph (QQuickFramebufferObject or updatePaintNode + QSGTexture) for the fastest, most maintainable rendering path inside your Qt app.

If you want, I can sketch the exact shm header, the Qt item skeleton, and the proxy write loop next.
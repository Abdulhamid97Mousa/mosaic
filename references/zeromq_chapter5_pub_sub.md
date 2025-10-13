ZeroMQ: Messaging for Many Applications — Chapter 5: Advanced Pub-Sub Patterns
=============================================================================

### Durable subscriber pattern

* Use a `zmq_proxy` or XPUB/XSUB device to mediate publishers and subscribers.
* Record subscriptions in the XPUB handler and filter them when bridging to PUB sockets.
* Snapshot state to new subscribers from a stable store before streaming live updates.

### Replay Protection

* Subscribers should track last-seen sequence identifier (monotonic counter).
* Publishers include sequence numbers in messages.
* On reconnect, subscriber requests replay lines from sequence n+1.

### Heartbeating

* PUB sockets send heartbeats (control frames) on dedicated channel.
* SUB sockets time out if no heartbeat seen and resubscribe or reconnect.

### Flow Control

* XPUB/XSUB devices should monitor `ZMQ_SNDHWM`/`ZMQ_RCVHWM`.
* Consider using `zmq_poll` to detect slow consumers; drop or spool messages when high-water marks hit.

### Reliable delivery recipe

1. Publisher writes every message to persistent storage.
2. Snapshotter service replies to subscriber with last consistent state.
3. Subscriber starts live feed and requests any missed frames (based on sequence).

### Security considerations

* Use CURVE (public-key) security for pub-sub endpoints when crossing trust boundaries.
* Authenticate subscriptions and enforce ACLs in the XPUB middleware.

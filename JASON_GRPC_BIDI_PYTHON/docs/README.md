# Jason ⇄ Python gRPC Bridge (Quick Start & Overview)

This document captures the steps we took to create the smallest possible bridge between the legacy Jason MAS project and a modern Python component using gRPC. Use it as a recipe to reproduce or extend the demo.

## Repo layout

```
JASON_GRPC_BIDI_PYTHON/
├── proto/agent_bridge.proto          # shared contract (bidirectional RPCs)
├── python_project/                   # Python gRPC server example
│   ├── agent_bridge_pb2*.py          # generated stub code
│   └── server.py                     # minimal handler storing percepts & returning actions
└── JASON_java_project/grpc-bridge-example/
    ├── build.gradle                  # adds protobuf + grpc plugins/deps
    ├── src/main/java/org/jason/...   # custom Jason environment + launcher
    └── src/main/resources/
        ├── grpc_bridge.mas2j         # MAS definition using the new env
        └── python_link.asl           # demo AgentSpeak agent
```

## Python side

1. Use the dedicated venv inside `JASON_GRPC_BIDI_PYTHON/.venv` (Python 3.11.14).
2. Install packages from `requirements.txt` (grpcio, grpcio-tools, protobuf) if needed.
3. Regenerate stubs after editing the proto:
   ```bash
   source .venv/bin/activate
   python -m grpc_tools.protoc -I proto \
       --python_out=python_project \
       --pyi_out=python_project \
       --grpc_python_out=python_project \
       proto/agent_bridge.proto
   ```
4. Start the toy server:
   ```bash
   source .venv/bin/activate
   python python_project/server.py --port 50051
   ```

The server keeps the latest percept per agent and returns heuristic actions (`recharge`, `cool_down`, etc.). Replace `_decide_action` with your own policy or forward to RL code.

## Java / Jason side

1. `settings.gradle` now includes the Gradle subproject `:grpc-bridge-example`.
2. `grpc-bridge-example/build.gradle` pulls in gRPC + protobuf plugins and points the proto source set to `../proto` so both languages share the same `.proto` file.
3. `GrpcBridgeEnvironment` implements two custom actions exposed to AgentSpeak:
   - `push_percept("json_or_text")` → invokes Python `PushPercept` RPC.
   - `request_action("context")` → gets the next action and injects it back as the percept `server_action(Action, Metadata)`.
4. `python_link.asl` shows how a Jason agent can call those actions within plans, while `grpc_bridge.mas2j` wires the environment/agent together.

### Building & running the demo

```bash
cd JASON_GRPC_BIDI_PYTHON/JASON_java_project
export JAVA_HOME=../.jdks/jdk-21.0.5+11
export PATH="$JAVA_HOME/bin:$PATH"
./gradlew :grpc-bridge-example:build     # compiles proto + java
./gradlew :grpc-bridge-example:run       # runs RunLocalMAS with grpc_bridge.mas2j
```

Run order for the integration test:
1. Start the Python server first (see above).
2. Run the Gradle `run` task. You should see logs similar to:
   ```
   [GrpcBridgeEnvironment] Connected to AgentBridge at localhost:50051
   [python_link] [Jason] python suggests: idle info:ctx_len=7;percept_len=0
   ```

## Next steps

- Replace the heuristic Python handler with real environment logic (RL policy, data collector, etc.).
- Extend the proto to cover streaming RPCs or agent-to-agent broadcasts if needed.
- Package the shared proto as its own module so other languages (C++, Go, etc.) can reuse it.
- Write MAS tests to exercise failure scenarios (Python unavailable, timeouts, etc.)
- Capture production instructions (systemd units, Dockerfiles) once the prototype hardens.

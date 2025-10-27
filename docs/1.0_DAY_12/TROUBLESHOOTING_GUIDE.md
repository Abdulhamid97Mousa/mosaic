# Troubleshooting Guide: Dynamic Agent Tabs & Image Naming

## Diagnostic Decision Trees

### Problem 1: Wrong Game Rendering in Dynamic Tabs

#### Diagnostic Flowchart

```mermaid
flowchart TD
    A["Dynamic tab shows<br/>wrong game?"] -->|Yes| B["Check database<br/>for game_id"]
    B -->|game_id=NULL| C["game_id not in payloads"]
    C -->|Root Cause| D["Worker doesn't emit game_id<br/>OR<br/>First payload missing game_id"]
    D -->|Fix| E["Pass game_id from<br/>run metadata to tab"]

    B -->|game_id=CLIFF_WALKING| F["game_id in database<br/>but tab shows wrong game"]
    F -->|Check logs| G["grep 'game_id' var/logs/"]
    G -->|game_id=None in logs| H["game_id not extracted<br/>from first payload"]
    H -->|Root Cause| I["First payload arrived<br/>without game_id field"]
    I -->|Fix| E

    G -->|game_id=CLIFF_WALKING in logs| J["game_id extracted correctly"]
    J -->|Check renderer| K["Check GridRenderer<br/>fallback logic"]
    K -->|Falls back to FROZEN_LAKE| L["RendererContext<br/>has game_id=None"]
    L -->|Root Cause| M["game_id not passed<br/>to RendererContext"]
    M -->|Fix| E

    style D fill:#ff6b6b
    style I fill:#ff6b6b
    style M fill:#ff6b6b
    style E fill:#6bcf7f
```

#### Quick Diagnostic Checks

**Step 1: Check if game_id exists in database**
```bash
sqlite3 var/telemetry/telemetry.sqlite \
  "SELECT DISTINCT game_id FROM steps WHERE game_id IS NOT NULL LIMIT 5;"
```

**What it reveals:**
- If result is empty → game_id never emitted by worker
- If result shows correct game → game_id is in database but not reaching tab

**Step 2: Check logs for game_id extraction**
```bash
grep -r "game_id" var/logs/ | grep -E "extracted|initialized|context" | head -20
```

**What it reveals:**
- If no matches → game_id extraction not logged (add logging)
- If shows "game_id=None" → First payload missing game_id
- If shows correct game_id → Problem is in renderer context

**Step 3: Check tab creation logs**
```bash
grep -r "Live.*agent\|_create_agent_tabs" var/logs/ | head -20
```

**What it reveals:**
- If no matches → Tab creation not logged (add logging)
- If shows tab created → Check if game_id was passed to constructor

**Step 4: Check renderer fallback**
```bash
grep -r "FROZEN_LAKE\|fallback" var/logs/ | head -20
```

**What it reveals:**
- If matches found → Renderer is falling back (game_id=None in context)
- If no matches → Renderer is using correct game_id

---

### Problem 2: Generic "Image.png" Instead of Timestamped Names

#### Diagnostic Flowchart

```mermaid
flowchart TD
    A["Frame shows 'Image.png'<br/>instead of timestamp?"] -->|Yes| B["Check database<br/>for frame_ref"]
    B -->|frame_ref=NULL| C["frame_ref not generated"]
    C -->|Root Cause| D["build_frame_reference()<br/>returns None"]
    D -->|Fix| E["Implement frame_ref<br/>generation in adapters"]

    B -->|frame_ref='Image.png'| F["Generic name in database"]
    F -->|Check adapter| G["Check build_frame_reference()<br/>implementation"]
    G -->|Returns None| H["Adapter not overriding<br/>build_frame_reference"]
    H -->|Root Cause| D
    H -->|Fix| E

    B -->|frame_ref='frames/...'| I["Timestamped name in database"]
    I -->|Check disk| J["ls var/records/run_id/frames/"]
    J -->|Files exist| K["Frame storage working"]
    K -->|Check UI| L["UI loading frames correctly"]

    J -->|No files| M["frame_ref in DB but<br/>files not on disk"]
    M -->|Root Cause| N["Frames not persisted<br/>before telemetry emitted"]
    N -->|Fix| O["Implement frame persistence<br/>before telemetry"]

    style D fill:#ff6b6b
    style N fill:#ff6b6b
    style E fill:#6bcf7f
    style O fill:#6bcf7f
    style K fill:#6bcf7f
```

#### Quick Diagnostic Checks

**Step 1: Check frame_ref in database**
```bash
sqlite3 var/telemetry/telemetry.sqlite \
  "SELECT DISTINCT frame_ref FROM steps WHERE frame_ref IS NOT NULL LIMIT 10;"
```

**What it reveals:**
- If empty → frame_ref is NULL (not generated)
- If shows "Image.png" → Generic fallback being used
- If shows "frames/..." → Timestamped names being generated

**Step 2: Check if frames exist on disk**
```bash
ls -la var/records/*/frames/ 2>/dev/null | head -20
```

**What it reveals:**
- If no directory → Frame storage not implemented
- If directory empty → Frames not being saved
- If files exist → Frame storage working

**Step 3: Check adapter implementation**
```bash
grep -r "build_frame_reference" gym_gui/core/adapters/ | head -20
```

**What it reveals:**
- If only shows "return None" → Not implemented
- If shows timestamp generation → Implementation exists
- If no matches → Method not overridden in adapters

**Step 4: Check telemetry emission logs**
```bash
grep -r "frame_ref\|frame.*generated" var/logs/ | head -20
```

**What it reveals:**
- If no matches → frame_ref generation not logged
- If shows "frame_ref=None" → Generation returning None
- If shows "frame_ref='frames/...'" → Generation working

---

## Data Model Analysis

### Complete Database Schema

```mermaid
erDiagram
    EPISODES ||--o{ STEPS : "contains"

    EPISODES {
        TEXT episode_id PK "Unique episode ID"
        REAL total_reward "Sum of rewards"
        INTEGER steps "Step count"
        INTEGER terminated "0/1"
        INTEGER truncated "0/1"
        BLOB metadata "JSON"
        TEXT timestamp "ISO-8601"
        TEXT agent_id "Agent ID"
        TEXT run_id "Run ID"
    }

    STEPS {
        TEXT episode_id FK "Episode reference"
        INTEGER step_index "0-based counter"
        INTEGER action "Action taken"
        BLOB observation "State JSON"
        REAL reward "Reward value"
        INTEGER terminated "0/1"
        INTEGER truncated "0/1"
        BLOB info "Metadata JSON"
        BLOB render_payload "Render data"
        TEXT timestamp "ISO-8601"
        TEXT agent_id "Agent ID"
        BLOB render_hint "Hints"
        TEXT frame_ref "Frame file ref"
        INTEGER payload_version "Schema version"
        TEXT run_id "Run ID"
    }
```

### Field-by-Field Mapping: Protobuf → Python → SQLite

```mermaid
graph LR
    subgraph Protobuf["Protobuf (RunStep)"]
        P1["episode_index: uint64"]
        P2["seq_id: uint64"]
        P3["action: int32"]
        P4["observation: bytes"]
        P5["reward: float"]
        P6["terminated: bool"]
        P7["truncated: bool"]
    end

    subgraph Python["Python (StepRecord)"]
        PY1["episode_id: str"]
        PY2["step_index: int"]
        PY3["action: int | None"]
        PY4["observation: Any"]
        PY5["reward: float"]
        PY6["terminated: bool"]
        PY7["truncated: bool"]
    end

    subgraph SQLite["SQLite (steps table)"]
        S1["episode_id: TEXT"]
        S2["step_index: INTEGER"]
        S3["action: INTEGER"]
        S4["observation: BLOB"]
        S5["reward: REAL"]
        S6["terminated: INTEGER"]
        S7["truncated: INTEGER"]
    end

    P1 -->|Reconstruct| PY1
    P2 -->|Direct| PY2
    P3 -->|Direct| PY3
    P4 -->|JSON| PY4
    P5 -->|Direct| PY5
    P6 -->|Direct| PY6
    P7 -->|Direct| PY7

    PY1 -->|Direct| S1
    PY2 -->|Direct| S2
    PY3 -->|Direct| S3
    PY4 -->|Serialize| S4
    PY5 -->|Direct| S5
    PY6 -->|0/1| S6
    PY7 -->|0/1| S7

    style P1 fill:#ffcccc
    style PY1 fill:#ffcccc
    style S1 fill:#ffcccc
```

**⚠️ Critical Transformation:**
- Protobuf `episode_index` (numeric) → Python `episode_id` (string)
- Reconstruction: `episode_id = f"{run_id}-ep{episode_index:04d}"`
- **Risk**: If reconstruction format changes, episodes become unretrievable

### Python Data Classes

#### StepRecord Fields

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `episode_id` | str | Reconstructed from episode_index | Episode grouping |
| `step_index` | int | seq_id from protobuf | Step ordering |
| `action` | int \| None | action from protobuf | Agent action |
| `observation` | Any | JSON-decoded observation | Environment state |
| `reward` | float | reward from protobuf | Step reward |
| `terminated` | bool | terminated from protobuf | Episode end flag |
| `truncated` | bool | truncated from protobuf | Episode truncation flag |
| `info` | Mapping | JSON-decoded info | Metadata |
| `timestamp` | datetime | Generated or from payload | When step occurred |
| `render_payload` | Any \| None | Grid/RGB data | Rendering data |
| `agent_id` | str \| None | From payload or default | Agent identifier |
| `render_hint` | Mapping \| None | Renderer hints | Rendering hints |
| `frame_ref` | str \| None | **⚠️ Usually None** | Frame file reference |
| `payload_version` | int | Schema version | Compatibility |
| `run_id` | str \| None | Training run ID | Run correlation |

#### EpisodeRollup Fields

| Field | Type | Source | Purpose |
|-------|------|--------|---------|
| `episode_id` | str | Reconstructed from episode_index | Episode identifier |
| `total_reward` | float | Sum of step rewards | Episode performance |
| `steps` | int | Step count | Episode length |
| `terminated` | bool | Episode end flag | Normal termination |
| `truncated` | bool | Episode truncation flag | Early stopping |
| `metadata` | Mapping | JSON-decoded metadata | Episode metadata |
| `timestamp` | datetime | When episode ended | Episode completion time |
| `agent_id` | str \| None | From payload or default | Agent identifier |
| `run_id` | str \| None | Training run ID | Run correlation |

### Telemetry Pipeline Architecture

```mermaid
graph TD
    A["Worker Process<br/>(JSONL)"] -->|episode_index| B["Trainer Proxy<br/>(gRPC)"]
    B -->|RunStep Proto| C["TelemetryAsyncHub<br/>_drain_loop"]
    C -->|Normalize| D["Payload Dict"]
    D -->|Extract| E["episode_id = f'{run_id}-ep{episode_index:04d}'"]
    E -->|Create| F["StepRecord"]
    F -->|Serialize| G["SQLite steps table"]
    G -->|Query| H["UI Display"]

    style A fill:#e1f5ff
    style B fill:#e1f5ff
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#ffcccc
    style F fill:#fff3e0
    style G fill:#fce4ec
    style H fill:#e8f5e9
```

**Transformation Points:**
1. **Worker → Proxy**: JSONL → Protobuf (episode_index)
2. **Proxy → Hub**: Protobuf → Python objects
3. **Hub → Normalization**: Add game_id, run_id, agent_id
4. **Normalization → StepRecord**: Reconstruct episode_id
5. **StepRecord → SQLite**: Serialize BLOB fields
6. **SQLite → UI**: Query and deserialize

---

## Telemetry Pipeline Architecture

### Queue and Buffer Configuration

```mermaid
graph LR
    A["Worker<br/>Emits"] -->|JSONL| B["Proxy<br/>Queue"]
    B -->|gRPC| C["Hub<br/>_drain_loop"]
    C -->|Normalize| D["RunBus<br/>Queue 64"]
    D -->|Subscribe| E["Controller<br/>Queue 64"]
    E -->|Route| F["Tab<br/>Buffer"]
    F -->|Render| G["UI<br/>Display"]

    H["Writer<br/>Thread"] -->|Batch| I["SQLite<br/>WAL"]

    D -.->|Overflow| J["Drop oldest<br/>event"]
    E -.->|Overflow| K["Drop oldest<br/>event"]

    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style J fill:#ff6b6b
    style K fill:#ff6b6b
    style I fill:#fce4ec
```

**Buffer Sizes:**
- RunBus queue: 64 events per subscriber
- Controller queue: 64 events
- Writer queue: 256-512 events (batched)
- Tab buffer: Depends on render throttle

**Overflow Policy:**
- When queue full: Drop oldest event
- Tracked in `_overflow` counter
- No feedback to producer (⚠️ Silent drops)

### Credit Backpressure Mechanism

```mermaid
sequenceDiagram
    participant Producer as Producer<br/>(Worker)
    participant CreditMgr as CreditManager
    participant RunBus as RunBus
    participant Tab as LiveTab

    Producer->>CreditMgr: Check credits
    CreditMgr-->>Producer: 200 credits available
    Producer->>RunBus: Publish STEP_APPENDED
    RunBus->>Tab: Event delivered
    Tab->>Tab: Process event
    Tab->>CreditMgr: Grant 100 credits<br/>(when queue < 50%)
    CreditMgr-->>Producer: Credits available

    Note over Producer,CreditMgr: ⚠️ Deadlock if tab<br/>not created yet!
```

**Credit Flow:**
1. Producer checks `CreditManager.get_credits(run_id, agent_id)`
2. If credits > 0: Consume 1 credit, publish event
3. If credits = 0: Wait or drop event
4. Tab grants credits when queue drops below 50%
5. **Problem**: Tab created AFTER first event, so first event has no credits

---

## Immediate Fixes

### Fix 1: Pass game_id to Dynamic Tabs

**File:** `gym_gui/ui/main_window.py`

In `_create_agent_tabs_for()`, extract game_id from run metadata:

```python
def _create_agent_tabs_for(self, run_id: str, agent_id: str, first_payload: dict) -> None:
    # Extract game_id from run metadata or first payload
    game_id = self._get_game_id_for_run(run_id)  # NEW
    
    # Pass to grid tab
    grid = AgentOnlineGridTab(
        run_id, 
        agent_id, 
        game_id=game_id,  # NEW
        renderer_registry=renderer_registry, 
        parent=self
    )
```

**File:** `gym_gui/ui/widgets/agent_online_grid_tab.py`

Update constructor:

```python
def __init__(
    self,
    run_id: str,
    agent_id: str,
    *,
    game_id: Optional[GameId] = None,  # NEW
    renderer_registry: Optional[RendererRegistry] = None,
    parent: Optional[QtWidgets.QWidget] = None,
) -> None:
    self._game_id = game_id  # Initialize from parameter
    # ... rest of init
```

### Fix 2: Implement Frame Reference Generation

**File:** `gym_gui/core/adapters/base.py`

Override `build_frame_reference()` in adapters:

```python
def build_frame_reference(self, render_payload: Any | None, state: StepState) -> str | None:
    """Generate timestamped frame reference."""
    if render_payload is None:
        return None
    
    # Generate timestamped filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    step_index = getattr(state, "step_index", 0)
    frame_ref = f"frames/{timestamp}_{step_index:06d}.png"
    return frame_ref
```

---

## Verification Steps

### After Fix 1:

```python
# In test or debug console
from gym_gui.ui.widgets.agent_online_grid_tab import AgentOnlineGridTab
from gym_gui.core.enums import GameId

tab = AgentOnlineGridTab("run1", "agent1", game_id=GameId.CLIFF_WALKING)
assert tab._game_id == GameId.CLIFF_WALKING  # Should pass
```

### After Fix 2:

```bash
# Check database for proper frame_ref values
sqlite3 var/telemetry/telemetry.sqlite \
  "SELECT frame_ref FROM steps WHERE frame_ref LIKE 'frames/%' LIMIT 5;"

# Should show: frames/20251021_093045_000001.png
```

---

## Why These Issues Occur: Root Cause Analysis

### Issue 1: game_id Context Loss

#### Why game_id is Lost

```mermaid
graph TD
    A["Run Configuration<br/>(game_id=CLIFF_WALKING)"] -->|Not passed| B["_create_agent_tabs_for()"]
    B -->|Relies on| C["First payload<br/>to extract game_id"]
    C -->|If missing| D["self._game_id = None"]
    D -->|Permanent| E["RendererContext(game_id=None)"]
    E -->|Fallback| F["GameId.FROZEN_LAKE"]

    style A fill:#6bcf7f
    style B fill:#ff6b6b
    style C fill:#ff6b6b
    style D fill:#ff6b6b
    style E fill:#ff6b6b
    style F fill:#ff6b6b
```

**Why this design is broken:**
1. **Metadata not propagated**: Run configuration stays in RunRegistry, not passed to tabs
2. **Late binding**: Tab tries to extract game_id from first payload instead of constructor
3. **No fallback**: If first payload missing game_id, no recovery mechanism
4. **Permanent state**: Once `self._game_id = None`, it never changes

**Why it worked before:**
- Payloads always included game_id (or adapters were different)
- First payload always arrived with game_id field
- Or frame storage was disabled, so game_id wasn't critical

### Issue 2: frame_ref Generation Incomplete

#### Why frame_ref is Always None

```mermaid
graph TD
    A["Adapter.render()"] -->|Should call| B["build_frame_reference()"]
    B -->|Currently returns| C["None"]
    C -->|Stored as| D["frame_ref=NULL"]
    D -->|UI shows| E["'Image.png' or missing"]

    F["Planned feature<br/>docs/1.0_DAY_5/"] -->|Not implemented| B

    style B fill:#ff6b6b
    style C fill:#ff6b6b
    style D fill:#ff6b6b
    style E fill:#ff6b6b
    style F fill:#ffd93d
```

**Why this design is incomplete:**
1. **Stub implementation**: `build_frame_reference()` returns None by default
2. **No frame storage**: No service to persist frames to disk
3. **No naming convention**: No timestamped filename generation
4. **No integration**: Frame storage not wired into telemetry pipeline

**Why it worked before:**
- Frame storage was manually implemented elsewhere
- Frames were saved by a different service
- Or frame display was disabled

---

## Long-term Solutions

### Solution 1: Metadata Pipeline Architecture

```mermaid
graph LR
    A["Run Config<br/>(game_id)"] -->|Pass through| B["RunRegistry"]
    B -->|Inject into| C["RunBus Event"]
    C -->|Include in| D["TelemetryEvent"]
    D -->|Extract in| E["LiveTelemetryController"]
    E -->|Pass to| F["Tab Constructor"]
    F -->|Initialize| G["self._game_id"]

    style A fill:#6bcf7f
    style B fill:#6bcf7f
    style C fill:#6bcf7f
    style D fill:#6bcf7f
    style E fill:#6bcf7f
    style F fill:#6bcf7f
    style G fill:#6bcf7f
```

**Implementation steps:**
1. Add `game_id` to `TelemetryEvent` dataclass
2. Extract from run metadata in `_drain_loop()`
3. Pass to `LiveTelemetryController` via event
4. Pass to tab constructor in `_create_agent_tabs_for()`

### Solution 2: Frame Storage Service

```mermaid
graph LR
    A["Adapter.render()"] -->|Call| B["build_frame_reference()"]
    B -->|Generate| C["Timestamped filename<br/>frames/20251021_093045_000001.png"]
    C -->|Save to| D["FrameStorageService"]
    D -->|Persist| E["var/records/run_id/frames/"]
    E -->|Return ref| F["frame_ref"]
    F -->|Store in| G["StepRecord"]
    G -->|Persist| H["SQLite"]

    style B fill:#6bcf7f
    style C fill:#6bcf7f
    style D fill:#6bcf7f
    style E fill:#6bcf7f
    style F fill:#6bcf7f
    style G fill:#6bcf7f
    style H fill:#6bcf7f
```

**Implementation steps:**
1. Create `FrameStorageService` singleton
2. Implement `build_frame_reference()` in all adapters
3. Generate timestamped filenames
4. Save frames before telemetry emission
5. Return frame_ref for storage

### Solution 3: Adapter Standardization

```mermaid
graph TD
    A["BaseAdapter"] -->|Define| B["build_frame_reference()"]
    B -->|Override in| C["GridAdapter"]
    B -->|Override in| D["RGBAdapter"]
    B -->|Override in| E["VideoAdapter"]

    C -->|Generate| F["frames/YYYYMMDD_HHMMSS_XXXXXX.png"]
    D -->|Generate| G["frames/YYYYMMDD_HHMMSS_XXXXXX.png"]
    E -->|Generate| H["frames/YYYYMMDD_HHMMSS_XXXXXX.mp4"]

    style B fill:#6bcf7f
    style C fill:#6bcf7f
    style D fill:#6bcf7f
    style E fill:#6bcf7f
    style F fill:#6bcf7f
    style G fill:#6bcf7f
    style H fill:#6bcf7f
```

**Implementation steps:**
1. Define interface in `BaseAdapter`
2. Override in all adapter subclasses
3. Generate format-specific filenames
4. Ensure consistent naming convention

---

## Related Documentation

- `ANALYSIS_DYNAMIC_AGENT_TABS_AND_IMAGE_NAMING.md` - Detailed analysis with diagrams
- `COMPREHENSIVE_TELEMETRY_ANALYSIS.md` - Complete telemetry system analysis
- `docs/1.0_DAY_5/` - Frame persistence roadmap
- `gym_gui/config/storage_profiles.yaml` - Storage configuration
- `spade_bdi_rl/core/telemetry.py` - Worker telemetry emission
- `gym_gui/services/trainer/streams.py` - TelemetryAsyncHub implementation
- `gym_gui/controllers/live_telemetry.py` - LiveTelemetryController implementation


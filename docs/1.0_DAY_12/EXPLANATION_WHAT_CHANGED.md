# Explanation: Why Image.png Appears Instead of Timestamped Names

## What You Observed

**Yesterday (2025-10-21_01-25_1.png):**
- Frame files had timestamped names
- Each frame was uniquely identified
- File paths were descriptive and traceable

**Today (Image.png):**
- Generic placeholder name
- Loss of temporal information
- Frames appear indistinguishable

---

## Why This Changed

### Root Cause: Frame Storage Implementation Status

The frame storage system in Gym GUI is **incomplete**. Looking at the codebase:

1. **Frame persistence is planned but not fully implemented**
   - See `docs/1.0_DAY_5/1.0_DAY_5_CURRENT_DESIGN_PROGRESS.md`
   - Status: "Open – JSONL telemetry persists, but no binary frame storage or `frame_ref`"

2. **No frame naming convention exists**
   - `build_frame_reference()` in `gym_gui/core/adapters/base.py` returns `None` by default
   - No adapter generates timestamped frame filenames
   - No frame storage service writes frames to disk

3. **Fallback to generic names**
   - When `frame_ref` is `None`, the UI may display a placeholder
   - "Image.png" is likely a fallback or default name
   - This suggests a recent change to handle missing frame_ref values

### Timeline of Changes

**What likely happened:**

1. **Initial implementation (yesterday)**
   - Someone manually created timestamped frame files
   - Or a temporary frame storage system was in place
   - Frame references were hardcoded or generated ad-hoc

2. **Recent refactoring (today)**
   - Frame storage was moved to a service-based architecture
   - The old ad-hoc frame naming was removed
   - New system relies on `frame_ref` field in telemetry
   - But `frame_ref` generation was never implemented
   - Result: `frame_ref = None` → fallback to "Image.png"

---

## Architecture Changes

### Old System (Yesterday)
```
Adapter → Render → Save to disk with timestamp
         ↓
         frame_2025-10-21_01-25_1.png
```

### New System (Today)
```
Adapter → build_frame_reference() → frame_ref (None)
         ↓
         Telemetry DB stores frame_ref=None
         ↓
         UI tries to load frame → fails → shows "Image.png"
```

---

## Why This Matters

### For Problem 1 (Context Issue)
The dynamic agent tabs were created as part of the new multi-agent telemetry system. They rely on:
- Proper `game_id` in telemetry (not implemented in workers)
- Correct `RendererContext` initialization (depends on game_id)
- Result: Context is broken when game_id is missing

### For Problem 2 (Image Naming)
The frame storage refactoring introduced:
- Dependency on `frame_ref` field
- But no implementation of frame_ref generation
- Result: Frames can't be loaded, fallback to placeholder

---

## What Needs to Happen

### Short-term (Fix the Regression)

1. **Restore frame_ref generation**
   - Implement `build_frame_reference()` in adapters
   - Generate timestamped filenames: `frames/YYYYMMDD_HHMMSS_XXXXXX.png`
   - Store in `var/records/{run_id}/frames/`

2. **Fix game_id propagation**
   - Pass game_id from run metadata to dynamic tabs
   - Don't rely on extracting from individual step payloads

### Long-term (Complete the Implementation)

1. **Frame Storage Service**
   - Implement `FrameStorageService` (planned in Day 5)
   - Handle PNG/WebP compression
   - Manage disk space with retention policies

2. **Telemetry Schema**
   - Make `game_id` required in worker telemetry
   - Add `frame_ref` as standard field
   - Document expected format

3. **Adapter Standardization**
   - All adapters must implement frame reference generation
   - Consistent naming convention across all environments

---

## Key Files Involved in the Change

| File | Role | Status |
|------|------|--------|
| `gym_gui/core/adapters/base.py` | Frame ref generation | ❌ Not implemented |
| `gym_gui/ui/widgets/agent_online_grid_tab.py` | Dynamic tab rendering | ⚠️ Broken context |
| `gym_gui/ui/main_window.py` | Tab creation | ⚠️ Missing game_id |
| `spade_bdi_rl/core/telemetry.py` | Worker telemetry | ⚠️ Missing game_id |
| `gym_gui/services/storage.py` | Storage profiles | ⚠️ Incomplete |

---

## Conclusion

The changes you're seeing are **not bugs** but rather **incomplete refactoring**:

- The old ad-hoc frame system was removed
- The new service-based system was partially implemented
- Frame reference generation was left unfinished
- Result: Regression from timestamped names to generic placeholders

This is a **known gap** documented in the Day 5 design journals. The fix requires implementing the frame storage service and ensuring game_id flows through the telemetry pipeline.


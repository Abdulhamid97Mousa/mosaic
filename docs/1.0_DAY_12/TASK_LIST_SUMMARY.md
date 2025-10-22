# Task List Summary: Telemetry & Live Tab System Fixes

## Overview

**Total Tasks**: 43 subtasks across 7 priority issues
**Estimated Effort**: 40-60 hours
**Risk Level**: Medium (architectural changes, but well-isolated)
**Dependencies**: None (can be done in priority order)

---

## Priority 1: Critical Deadlock Issues (10 tasks)

### P1.1-P1.5: Credit Backpressure Chicken-and-Egg Deadlock

**Problem**: Producer won't publish until credits arrive, but credits only sent after tab created.

**Tasks**:
- [ ] P1.1: Verify credit initialization timing
- [ ] P1.2: Trace credit grant flow from tab creation
- [ ] P1.3: Implement pre-tab credit grant mechanism
- [ ] P1.4: Add diagnostic logging for credit flow
- [ ] P1.5: Test credit backpressure with first-event scenario

**Acceptance Criteria**:
- Credits initialized before first event
- No deadlock when tab is created after first event
- All events published to UI
- Logs show credit flow clearly

**Estimated Effort**: 8-12 hours

---

### P1.6-P1.10: RunBus vs Qt-Signal Delivery Split

**Problem**: Two paths to UI (Qt signals and RunBus). One may be disabled.

**Tasks**:
- [ ] P1.6: Verify RunBus publishing in _drain_loop
- [ ] P1.7: Verify LiveTelemetryController subscription to RunBus
- [ ] P1.8: Check for Qt signal path vs RunBus path
- [ ] P1.9: Add RunBus publishing diagnostics
- [ ] P1.10: Test RunBus event delivery end-to-end

**Acceptance Criteria**:
- RunBus publishing confirmed in _drain_loop
- Controller receives all events from RunBus
- No events dropped due to path issues
- Diagnostics show event flow clearly

**Estimated Effort**: 6-10 hours

---

## Priority 2: Data Model Mismatches (10 tasks)

### P2.1-P2.5: Episode Identity Mismatch

**Problem**: Protobuf carries episode_index, Python uses episode_id. Mismatch causes episodes to vanish.

**Tasks**:
- [ ] P2.1: Audit episode_id vs episode_index usage
- [ ] P2.2: Verify episode_id generation consistency
- [ ] P2.3: Check episode_id in protobuf messages
- [ ] P2.4: Verify episode_id in database queries
- [ ] P2.5: Test episode identity end-to-end

**Acceptance Criteria**:
- episode_id consistent across all components
- No episodes lost in conversion
- Database queries work correctly
- End-to-end test passes

**Estimated Effort**: 8-12 hours

---

### P2.6-P2.10: Agent ID Filtering Mismatch

**Problem**: Proxy injects agent_id, but tab key uses different format. Events silently dropped.

**Tasks**:
- [ ] P2.6: Verify agent_id extraction from payloads
- [ ] P2.7: Verify agent_id in tab creation
- [ ] P2.8: Check agent_id in LiveTelemetryController routing
- [ ] P2.9: Add agent_id validation and logging
- [ ] P2.10: Test agent_id routing with multiple agents

**Acceptance Criteria**:
- agent_id consistent across all components
- No events dropped due to agent_id mismatch
- Multiple agents have separate tabs
- Logs show routing decisions clearly

**Estimated Effort**: 6-10 hours

---

## Priority 3: Buffer & Display Issues (13 tasks)

### P3.1-P3.5: Buffer/Backpressure Thresholds

**Problem**: UI buffers overflow silently. Tab counter doesn't advance.

**Tasks**:
- [ ] P3.1: Audit buffer thresholds and overflow handling
- [ ] P3.2: Verify credit consumption vs buffer capacity
- [ ] P3.3: Check throttle application to tabs
- [ ] P3.4: Add overflow metrics and visibility
- [ ] P3.5: Test buffer overflow scenario

**Acceptance Criteria**:
- Buffer thresholds properly configured
- Credits aligned with buffer capacity
- Overflow events visible to user
- No silent drops

**Estimated Effort**: 6-10 hours

---

### P3.6-P3.9: game_id Context Propagation

**Problem**: Dynamic tabs receive game_id=None, fall back to FROZEN_LAKE.

**Tasks**:
- [ ] P3.6: Extract game_id from run metadata
- [ ] P3.7: Pass game_id to tab constructors
- [ ] P3.8: Fix RendererContext initialization
- [ ] P3.9: Test game_id propagation end-to-end

**Acceptance Criteria**:
- game_id extracted from run metadata
- Passed to tab constructors
- Correct game rendered
- No fallback to FROZEN_LAKE

**Estimated Effort**: 4-6 hours

---

### P3.10-P3.13: Frame Reference Generation

**Problem**: frame_ref is None because build_frame_reference() returns None.

**Tasks**:
- [ ] P3.10: Implement frame_ref generation in adapters
- [ ] P3.11: Wire frame_ref into telemetry pipeline
- [ ] P3.12: Implement frame persistence
- [ ] P3.13: Test frame reference end-to-end

**Acceptance Criteria**:
- Timestamped frame filenames generated
- Frames persisted to disk
- frame_ref included in telemetry
- UI can load frames

**Estimated Effort**: 8-12 hours

---

## Implementation Order

### Phase 1: Diagnostics (2-4 hours)
1. Run diagnostic checks
2. Identify which issues are present
3. Prioritize based on findings

### Phase 2: Priority 1 Fixes (14-22 hours)
1. Implement pre-tab credit grant (P1.1-P1.5)
2. Verify RunBus publishing (P1.6-P1.10)
3. Test both fixes

### Phase 3: Priority 2 Fixes (14-22 hours)
1. Fix episode identity (P2.1-P2.5)
2. Fix agent_id routing (P2.6-P2.10)
3. Test both fixes

### Phase 4: Priority 3 Fixes (18-28 hours)
1. Fix buffer/backpressure (P3.1-P3.5)
2. Fix game_id propagation (P3.6-P3.9)
3. Implement frame references (P3.10-P3.13)
4. Test all fixes

### Phase 5: Integration Testing (4-8 hours)
1. End-to-end testing
2. Multi-agent testing
3. Stress testing
4. Regression testing

---

## Risk Assessment

| Issue | Risk | Mitigation |
|-------|------|-----------|
| Credit deadlock | High | Pre-initialize credits, add logging |
| RunBus split | Medium | Verify both paths, add diagnostics |
| Episode identity | Medium | Audit usage, add validation |
| Agent ID mismatch | Low | Standardize format, add logging |
| Buffer overflow | Low | Align thresholds, add visibility |
| game_id context | Low | Extract from metadata, pass to tabs |
| Frame references | Low | Implement generation, add persistence |

---

## Success Metrics

- [ ] All 7 issues resolved
- [ ] No regressions in existing functionality
- [ ] Performance impact < 5%
- [ ] All tests pass
- [ ] Logs clearly show data flow
- [ ] User can see dynamic tabs with data
- [ ] No deadlocks or silent drops

---

## Documentation

- `COMPREHENSIVE_TELEMETRY_ANALYSIS.md` - Detailed analysis of all issues
- `IMPLEMENTATION_GUIDE.md` - Step-by-step implementation instructions
- `TROUBLESHOOTING_GUIDE.md` - Diagnostic and troubleshooting steps
- `ANALYSIS_DYNAMIC_AGENT_TABS_AND_IMAGE_NAMING.md` - Original analysis
- `EXPLANATION_WHAT_CHANGED.md` - Why image naming changed

---

## Next Steps

1. Review this task list
2. Run diagnostic checks from Phase 1
3. Identify which issues are present
4. Start with Priority 1 fixes
5. Test after each phase
6. Document findings and solutions


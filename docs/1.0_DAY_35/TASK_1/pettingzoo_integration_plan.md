# PettingZoo Integration Plan

**Date:** 2025-11-27 (Updated: 2025-11-28)
**Status:** Phase 3 ✅ COMPLETE (Board Game Rendering)
**Goal:** Integrate PettingZoo as the Multi-Agent environment backend for gym_gui

---

## Current Status Summary

### ✅ Completed

#### Phase 1: Core Infrastructure
- [x] Created `gym_gui/core/pettingzoo_enums.py` with environment IDs, families, API types
- [x] Added PETTINGZOO to EnvironmentFamily enum
- [x] Added `pettingzoo` to requirements.txt

#### Phase 2: GUI Tab Widget
- [x] Created `gym_gui/ui/widgets/multi_agent_tab.py` with three subtabs
- [x] Updated `control_panel.py` to use MultiAgentTab
- [x] Connected signals through to MainWindow
- [x] Created PettingZoo game documentation

#### Phase 3: Board Game Rendering ✅ NEW
- [x] Created `gym_gui/rendering/strategies/board_game.py` - Consolidated strategy
- [x] Updated `render_tabs.py` - Uses BoardGameRendererStrategy in Grid tab
- [x] Updated `main_window.py` - Direct handler connections
- [x] Removed standalone board widget files (consolidated)
- [x] Fixed game detection for adapter payloads (game_type field)
- [x] Dynamic tab titles ("Grid - Chess", "Grid - Go", "Grid - Connect Four")

### ⏳ Pending

#### Phase 4: Training Support
- [ ] Add worker selection to HumanVsAgentTab
- [ ] Create training form for PettingZoo Classic
- [ ] Implement self-play PPO algorithm
- [ ] Policy loading for Human vs Agent

---

## Architecture

### Board Game Rendering (Implemented)

```
RenderTabs
└── Grid Tab (dynamic title)
    └── BoardGameRendererStrategy
        └── _BoardGameWidget (QStackedWidget)
            ├── _ChessBoardRenderer (interactive)
            ├── _ConnectFourBoardRenderer (interactive)
            └── _GoBoardRenderer (interactive)
```

### Key Design Decisions

1. **State-Based Rendering**: Send game state to Qt, not just pixels - enables mouse interaction
2. **Consolidated Strategy**: Single file for all board games instead of separate widgets
3. **Grid Tab Integration**: Board games render in Grid tab, not separate PettingZoo tab
4. **Lazy Loading**: Renderers created on-demand
5. **Game Detection**: Detects from `game_type` field or content keys (`fen`, etc.)

---

## Files Status

### Created/Updated (Phase 3)

| File | Status | Purpose |
|------|--------|---------|
| `gym_gui/rendering/strategies/board_game.py` | ✅ Created | Consolidated board game rendering |
| `gym_gui/ui/widgets/render_tabs.py` | ✅ Updated | Uses BoardGameRendererStrategy |
| `gym_gui/ui/main_window.py` | ✅ Updated | Direct handler connections |
| `gym_gui/ui/handlers/__init__.py` | ✅ Updated | Removed PettingZooClassicHandler |

### Removed (Consolidated)

| File | Reason |
|------|--------|
| `gym_gui/ui/widgets/chess_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/go_board.py` | Merged into board_game.py |
| `gym_gui/ui/widgets/connect_four_board.py` | Merged into board_game.py |
| `gym_gui/ui/environments/multi_agent_env/pettingzoo/pettingzoo_tab.py` | Replaced by Grid tab |
| `gym_gui/ui/handlers/pettingzoo_classic_handler.py` | Direct connections |

### Kept (In Use)

| File | Purpose |
|------|---------|
| `gym_gui/core/pettingzoo_enums.py` | Environment and API type enums |
| `gym_gui/ui/widgets/multi_agent_tab.py` | Multi-Agent tab with subtabs |
| `gym_gui/ui/handlers/chess_handlers.py` | Chess move handling |
| `gym_gui/ui/handlers/go_handlers.py` | Go move handling |
| `gym_gui/ui/handlers/connect_four_handlers.py` | Connect Four handling |

---

## Game Detection Logic

```python
def get_game_from_payload(payload):
    # Check content keys
    if "chess" in payload or "fen" in payload:
        return GameId.CHESS
    if "connect_four" in payload:
        return GameId.CONNECT_FOUR
    if "go" in payload:
        return GameId.GO

    # Check game_type field (from adapter to_dict())
    game_type = payload.get("game_type")
    if game_type == "chess":
        return GameId.CHESS
    if game_type == "connect_four":
        return GameId.CONNECT_FOUR
    if game_type == "go":
        return GameId.GO
```

---

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETE
- [x] Environment enums
- [x] Requirements update

### Phase 2: GUI Tab Widget ✅ COMPLETE
- [x] MultiAgentTab with subtabs
- [x] Signal connections

### Phase 3: Board Game Rendering ✅ COMPLETE
- [x] BoardGameRendererStrategy
- [x] Grid tab integration
- [x] Game detection from payloads
- [x] Interactive mouse handling

### Phase 4: Training Support ⏳ PENDING
- [ ] Worker selection in HumanVsAgentTab
- [ ] Training configuration form
- [ ] Self-play algorithm
- [ ] Policy loading

---

## Log Codes

| Code | Level | Description |
|------|-------|-------------|
| `LOG750` | INFO | Multi-agent environment load requested |
| `LOG751` | INFO | Multi-agent environment loaded successfully |
| `LOG752` | ERROR | Multi-agent environment load failed |
| `LOG753` | INFO | Multi-agent policy load requested |
| `LOG754` | INFO | Multi-agent game start requested |
| `LOG755` | INFO | Multi-agent reset requested |
| `LOG756` | DEBUG | Multi-agent action submitted |
| `LOG757` | INFO | Multi-agent training requested |
| `LOG758` | INFO | Multi-agent evaluation requested |
| `LOG759` | WARNING | Action attempted without environment |

---

## Next Steps (Phase 4)

1. **Update HumanVsAgentTab UI**
   - Add worker selection dropdown
   - Add "Configure Training" and "Train Agent" buttons

2. **Create Training Form**
   - `gym_gui/ui/widgets/pettingzoo_classic_train_form.py`
   - Self-play PPO configuration

3. **Implement Self-Play Algorithm**
   - `3rd_party/cleanrl_worker/cleanrl/cleanrl/ppo_pettingzoo_classic_selfplay.py`

4. **Policy Loading**
   - `gym_gui/ui/widgets/pettingzoo_classic_policy_form.py`
   - Discover and load trained policies

---

## References

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [PettingZoo GitHub](https://github.com/Farama-Foundation/PettingZoo)
- [CleanRL + PettingZoo Tutorial](https://pettingzoo.farama.org/tutorials/cleanrl/)

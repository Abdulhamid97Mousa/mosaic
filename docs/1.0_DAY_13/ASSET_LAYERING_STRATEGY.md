# Asset Layering Strategy for Grid-Based Environments

**Date**: October 22, 2025  
**Status**: Implemented  
**Applies to**: FrozenLake, FrozenLake-v2, CliffWalking

## Overview

The asset layering strategy is a rendering architecture that composes multiple asset layers to create the final visual representation of grid tiles. Instead of selecting a single monolithic asset per cell, the system:

1. Identifies a **base layer** (e.g., ice, mountain background)
2. Identifies zero or more **overlay layers** (e.g., goal marker, cliff hazard, agent stool)
3. **Composites** layers from bottom to top using transparent alpha blending

This approach provides:
- **Consistency**: Uniform base across cells reduces asset count
- **Flexibility**: Easy to add new overlays without creating new base tiles
- **Clarity**: Clear separation between "what is this tile?" and "what's on top?"
- **Maintainability**: Asset management logic centralized in `AssetManager`

## Architecture

### Asset Modules

**File**: `gym_gui/rendering/assets.py`

Each environment has a dedicated asset class:

#### FrozenLakeAssets
```python
class FrozenLakeAssets:
    ICE = "ice.png"              # Base layer
    HOLE = "hole.png"            # Overlay: hazard
    CRACKED_HOLE = "cracked_hole.png"  # Overlay: terminal hazard
    GOAL = "goal.png"            # Overlay: target
    STOOL = "stool.png"          # Overlay: start position
    AGENT_UP/DOWN/LEFT/RIGHT     # Actor sprites
```

**Key Method**: `get_tile_layers(cell_value, row, col, terminated)`
- **Input**: Cell character ('S', 'F', 'H', 'G'), position, episode state
- **Output**: Ordered list of asset filenames
- **Example**: `['ice.png', 'hole.png']` for a hole tile

#### CliffWalkingAssets
```python
class CliffWalkingAssets:
    MOUNTAIN_BG1/BG2 = "mountain_bg*.png"  # Base layers (alternating for visual variety)
    MOUNTAIN_CLIFF = "mountain_cliff.png"  # Overlay: cliff hazard
    MOUNTAIN_NEAR_CLIFF1/2 = "mountain_near-cliff*.png"  # Overlays: near-cliff warnings
    STOOL = "stool.png"          # Overlay: start position
    COOKIE = "cookie.png"        # Overlay: goal
    AGENT_UP/DOWN/LEFT/RIGHT     # Actor sprites
```

**Key Methods**:
- `get_tile_layers(cell_value, row, col)`: Returns ordered list of asset layers
- `_base_layer(cell, row, col)`: Determines background (varies by row for visual richness)
- `_overlay_layer(cell, row, col)`: Determines foreground marker (cliff, goal, stool)

### Renderer Integration

**File**: `gym_gui/rendering/strategies/grid.py`

The `_GridRenderer._create_cell_pixmap()` method implements layer composition:

```python
# For FrozenLake (post-refactor)
elif self._current_game in (GameId.FROZEN_LAKE, GameId.FROZEN_LAKE_V2):
    terminated_for_hole = terminated and is_actor_cell and cell_value.upper() == "H"
    layer_names = FrozenLakeAssets.get_tile_layers(cell_value, terminated=terminated_for_hole)
    pixmap = None
    for asset_name in layer_names:
        layer_pixmap = self._asset_manager.get_pixmap(asset_name)
        if layer_pixmap is None:
            continue
        if pixmap is None:
            pixmap = layer_pixmap  # First layer becomes base
        else:
            pixmap = self._composite_pixmaps(pixmap, layer_pixmap)  # Stack overlay

# For CliffWalking
elif self._current_game == GameId.CLIFF_WALKING:
    layer_names = CliffWalkingAssets.get_tile_layers(cell_value, row, col)
    pixmap = None
    for asset_name in layer_names:
        layer_pixmap = self._asset_manager.get_pixmap(asset_name)
        if layer_pixmap is None:
            continue
        if pixmap is None:
            pixmap = layer_pixmap
        else:
            pixmap = self._composite_pixmaps(pixmap, layer_pixmap)
```

### Pixmap Composition

The `_composite_pixmaps(base, overlay)` method:

```python
def _composite_pixmaps(base: QtGui.QPixmap, overlay: QtGui.QPixmap) -> QtGui.QPixmap:
    result = QtGui.QPixmap(base.size())
    result.fill(QtCore.Qt.GlobalColor.transparent)
    
    painter = QtGui.QPainter(result)
    painter.drawPixmap(0, 0, base)  # Draw base layer
    
    # Center the overlay on the base
    x_offset = (base.width() - overlay.width()) // 2
    y_offset = (base.height() - overlay.height()) // 2
    painter.drawPixmap(x_offset, y_offset, overlay)  # Draw overlay on top
    
    painter.end()
    return result
```

## Examples

### FrozenLake Tile Rendering

| Cell | Layers | Visual Result |
|------|--------|---------------|
| `S` (Start) | `['ice.png', 'stool.png']` | Ice with stool icon |
| `F` (Frozen) | `['ice.png']` | Plain ice |
| `H` (Hole) | `['ice.png', 'hole.png']` | Ice with hole icon |
| `H` (Hole, terminated) | `['ice.png', 'cracked_hole.png']` | Ice with cracked hole icon (terminal crash effect) |
| `G` (Goal) | `['ice.png', 'goal.png']` | Ice with goal flag |

### CliffWalking Tile Rendering

| Position | Cell | Layers | Visual Result |
|----------|------|--------|---------------|
| (3, 0) | `x` | `['mountain_bg2.png', 'stool.png']` | Mountain with stool (start) |
| (3, 1-10) | `C` | `['mountain_bg1/2.png', 'mountain_cliff.png']` | Mountain with cliff hazard |
| (3, 11) | `T` | `['mountain_bg1.png', 'cookie.png']` | Mountain with cookie (goal) |
| (0-2, any) | `.` | `['mountain_bg1/2.png']` | Base mountain layer only |

## Benefits

### Code Clarity
- **Assets module**: Centralized layer definitions
- **Renderer**: Generic layer composition logic
- **No duplicates**: Single "ice" base, not multiple "ice_with_*" tiles

### Visual Consistency
- All FrozenLake cells share the same ice base texture
- All CliffWalking cells use consistent mountain backgrounds
- Overlays consistently centered and centered on bases

### Extensibility
To add a new tile type (e.g., "slippery ice" warning):
1. Add asset: `SLIPPERY_ICE = "slippery_ice.png"`
2. Update `_overlay_layer()` to return it for specific conditions
3. Renderer automatically composes it with the base

No need to create new combined assets!

### Performance
- Asset caching: Single "ice.png" loaded once, reused for all ice tiles
- Composition happens on first render, results cached in Qt's scene graph
- Tile size: 120×120px with caching makes rendering smooth even with many cells

## Testing

### Visual Verification Checklist

- [ ] FrozenLake tiles display correctly:
  - Start tile shows stool icon
  - Holes show hole icon (or cracked icon when crashed)
  - Goal shows flag icon
  - Safe tiles show plain ice

- [ ] CliffWalking tiles display correctly:
  - Start tile shows stool on mountain
  - Cliff tiles show cliff overlay
  - Goal tile shows cookie on mountain
  - Background varies (BG1/BG2) for visual interest

- [ ] Overlay centering:
  - All overlays centered horizontally and vertically on base
  - No partial overlays clipped at edges
  - Overlay size proportional to tile

## Current Status

✅ **Implemented**:
- CliffWalkingAssets: Full layering support with `get_tile_layers()`
- FrozenLakeAssets: Layering support added (via `get_tile_layers()` method)
- GridRendererStrategy: Both FrozenLake and CliffWalking use layering consistently
- Pixmap composition: Working with proper alpha blending and centering

✅ **Verified**:
- Asset files exist in `gym_gui/assets/toy_text_images/`
- Layer composition logic correct
- No Codacy issues introduced

## Related Files

- `gym_gui/rendering/assets.py` - Asset definitions and layer selection logic
- `gym_gui/rendering/strategies/grid.py` - Layer composition in renderer
- `gym_gui/rendering/registry.py` - RendererRegistry instantiation
- `gym_gui/rendering/interfaces.py` - RendererStrategy protocol definition

## Future Enhancements

- **TaxiAssets layering**: Extend Taxi to use layering for border pieces
- **Layer presets**: Named layer combinations (e.g., "ice_cracked" = ICE + CRACKED_HOLE)
- **Animation layers**: Add temporal variation (e.g., "ice_wet_0", "ice_wet_1", ...) for animated tiles
- **Accessibility**: Add layer for high-contrast overlays for colorblind users

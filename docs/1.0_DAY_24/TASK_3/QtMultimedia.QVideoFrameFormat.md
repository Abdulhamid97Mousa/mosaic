# QtMultimedia.QVideoFrameFormat (PySide6)

> Reference: [Qt for Python - QVideoFrameFormat](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QVideoFrameFormat.html)

`QVideoFrameFormat` describes the format of frames delivered to a video sink. It captures how pixels are laid out, how large each frame is, and which color interpretations should be applied so that a renderer can display the stream correctly.

## Why it matters

- Controls the pixel layout (`pixelFormat()`), size (`frameSize()`), and visible region (`viewport()`) used by sinks (`QVideoSink`, `QVideoWidget`, etc.).
- Encodes color space metadata (transfer, range, and YCbCr space) that downstream renderers need for correct tone mapping and HDR playback.
- Stores stream-level hints such as scan-line order, mirroring, and rotation so that sinks can compensate for camera orientation without copying the frame buffer.

## Construction patterns

| Factory | Typical use |
| --- | --- |
| `QVideoFrameFormat()` | Creates an invalid/null format. Use `isValid()` before consuming. |
| `QVideoFrameFormat(other)` | Copy-construct from an existing format. |
| `QVideoFrameFormat(size, pixelFormat)` | Specify the frame dimensions (`QSize`) and a `PixelFormat` enum value. |

> **Tip:** Changing the frame size with `setFrameSize()` resets the viewport to the full frame. Adjust the viewport afterwards if you need letterboxing.

## Core getters and setters

| Getter | Setter | Notes |
| --- | --- | --- |
| `pixelFormat()` | – | Underpins all other metadata. Use `planeCount()` to query plane layout. |
| `frameSize()/frameWidth()/frameHeight()` | `setFrameSize(width, height)` | Dimensions of the raw frame buffer. |
| `viewport()` | `setViewport(QRect)` | Displayed sub-region (defaults to full frame). |
| `frameRate()` *(deprecated)* | `setFrameRate(float)` *(deprecated)* | Prefer `streamFrameRate()`/`setStreamFrameRate()` for negotiated rates. |
| `streamFrameRate()` | `setStreamFrameRate(float)` | Delivery cadence after pipeline negotiation. |
| `colorSpace()` | `setColorSpace(ColorSpace)` | Replaces legacy `yCbCrColorSpace()`. |
| `colorTransfer()` | `setColorTransfer(ColorTransfer)` | HDR metadata, e.g., PQ or HLG. |
| `colorRange()` | `setColorRange(ColorRange)` | Differentiates studio-range vs full-range content. |
| `maxLuminance()` | `setMaxLuminance(float)` | Upper bound for HDR tone mapping. |
| `isMirrored()` | `setMirrored(bool)` | Horizontal flip after rotation; useful for front cameras. |
| `rotation()` | `setRotation(Rotation)` | Clockwise rotation applied before mirroring. |
| `scanLineDirection()` | `setScanLineDirection(Direction)` | Top-to-bottom vs bottom-to-top buffers. |

Additional helpers: `swap(other)`, `updateUniformData()` (feeds shaders), `fragmentShaderFileName()`, `vertexShaderFileName()`.

## Pixel format families

`QVideoFrameFormat.PixelFormat` covers both RGB(A) layouts and YUV sampling schemes. Grouping them by usage makes the list easier to scan.

### RGBA/BGRA/XRGB (8-bit per channel)

| Constant | Description |
| --- | --- |
| `Format_Invalid` | Placeholder for invalid frames. |
| `Format_ARGB8888`, `Format_ARGB8888_Premultiplied` | 8-bit ARGB; premultiplied variant is alpha-multiplied. |
| `Format_BGRA8888`, `Format_BGRA8888_Premultiplied` | Byte order `B G R A`. |
| `Format_ABGR8888`, `Format_XBGR8888` | `A/B/G/R` ordering, with `X` as unused byte. |
| `Format_RGBA8888`, `Format_RGBX8888`, `Format_BGRX8888`, `Format_XRGB8888` | Variants with or without alpha; `X` marks padding. |

### Packed & planar YUV (8-bit unless stated)

| Constant | Layout |
| --- | --- |
| `Format_AYUV`, `Format_AYUV_Premultiplied` | 32-bit packed AYUV (alpha + YUV). |
| `Format_YUV420P` | Planar 4:2:0 (Y + subsampled U/V planes). |
| `Format_YUV422P` | Planar 4:2:2 (Y + half-width U/V). |
| `Format_YV12` | Planar 4:2:0 with V and U swapped. |
| `Format_UYVY`, `Format_YUYV` | Packed macropixels storing two luma samples plus shared chroma. |
| `Format_NV12`, `Format_NV21` | Semi-planar 4:2:0 with interleaved UV or VU. |
| `Format_IMC1`-`Format_IMC4` | Intel planar variants matching 4:2:0/4:2:0 VU layouts. |
| `Format_YUV420P10`, `Format_P010`, `Format_P016` | High-bit-depth 4:2:0 (10-bit or 16-bit stored in 16-bit words). |

### Monochrome, compressed, textures

| Constant | Usage |
| --- | --- |
| `Format_Y8`, `Format_Y16` | 8-bit or 16-bit grayscale. |
| `Format_Jpeg` | Compressed JPEG bitstream. |
| `Format_SamplerExternalOES` | Android external OES texture. |
| `Format_SamplerRect` | macOS rectangle texture (`GL_TEXTURE_RECTANGLE`). |

Use `pixelFormatToString()` for logging and `imageFormatFromPixelFormat()` when mapping to `QImage` (note: YUV formats generally lack direct `QImage` support).

## Orientation & scan direction

| Enum | Values |
| --- | --- |
| `Direction` | `TopToBottom`, `BottomToTop`. Set with `setScanLineDirection()`. |
| `Rotation` | Exposed via `rotation()`/`setRotation()`. Values mirror `QtVideo::Rotation` (None, 90, 180, 270). |
| Mirroring | `isMirrored()`/`setMirrored()` perform a horizontal flip after rotation. |

## Color metadata enums

### `ColorSpace` (PySide6 6.4+)

| Constant | Use case |
| --- | --- |
| `ColorSpace_Undefined` | No metadata; assume sRGB. |
| `ColorSpace_BT601` | Legacy SD video. |
| `ColorSpace_BT709` | Default for HD content. |
| `ColorSpace_AdobeRgb` | Wide-gamut still-image workflows. |
| `ColorSpace_BT2020` | HDR / UHD pipelines. |

### `ColorTransfer`

| Constant | Notes |
| --- | --- |
| `ColorTransfer_Unknown` | No curve provided. |
| `ColorTransfer_BT709` / `ColorTransfer_BT601` | Standard dynamic range EOTFs. |
| `ColorTransfer_Linear` | Linear light values (rare). |
| `ColorTransfer_Gamma22`, `ColorTransfer_Gamma28` | Explicit gamma curves. |
| `ColorTransfer_ST2084` | Perceptual Quantizer (PQ) HDR. |
| `ColorTransfer_STD_B67` | Hybrid Log-Gamma (HLG) HDR. |

### `ColorRange`

| Constant | Range |
| --- | --- |
| `ColorRange_Unknown` | Not specified. |
| `ColorRange_Video` | Studio/video range (for 8-bit, Y = 16-235 and Cb/Cr = 16-240). |
| `ColorRange_Full` | Full 0 to (2^depth - 1) range. |

### Legacy `YCbCrColorSpace`

Deprecated in favor of `ColorSpace`, but still present for raw YUV formats. Values include `YCbCr_Undefined`, `YCbCr_BT601`, `YCbCr_BT709`, `YCbCr_BT2020`, plus legacy xvYCC and JPEG variants.

## Utility functions

- `planeCount()` determines how many planes the chosen pixel format exposes. RGB formats report 1; YUV varies between 1 and 3.
- `pixelFormatFromImageFormat(QImage.Format)` helps when you receive frames via `QImage` and need to create matching `QVideoFrameFormat` instances.
- `updateUniformData()` fills shader uniform buffers (used internally by Qt's rendering stages; rarely invoked manually).

## Usage guidelines for this project

1. **Negotiate formats early:** When instantiating `QVideoSink`, check `supportedPixelFormats()` and construct a `QVideoFrameFormat` that matches the sinks strongest profile to avoid runtime conversion.
2. **Honor orientation metadata:** When processing camera frames, read `rotation()` and `isMirrored()` before feeding frames into custom render or image-processing code.
3. **Respect HDR metadata:** If `colorTransfer()` returns `ST2084` or `STD_B67`, ensure downstream shaders or tone-mappers understand HDR EOTFs; otherwise fall back to SDR-safe conversion.
4. **Viewport for letterboxing:** Use `setViewport()` to represent pillarboxed or letterboxed streams without resizing the underlying buffer, which avoids copying.
5. **Consistency across modules:** Store negotiated formats in shared DTOs so telemetry, recording, and on-screen rendering maintain the same interpretation of each frame.

## Related APIs

- `QVideoSink` / `QMediaCaptureSession` consumers that emit `QVideoFrame`s using this format.
- `QCameraDevice::videoFormats()` lists available `QVideoFrameFormat` combinations per camera.
- `QImage` / `QVideoFrame` conversions use `QVideoFrame::toImage()` when you need CPU-side image processing; note dropped metadata such as `colorTransfer()` must be reapplied if you return to video sinks.

---

*Last reviewed: Day 24 (Qt Multimedia research sprint). Update this file when new pixel formats or color metadata enums land in future Qt releases.*

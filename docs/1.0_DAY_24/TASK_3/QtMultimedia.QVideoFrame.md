PySide6.QtMultimedia.QVideoFrame
class QVideoFrame
The QVideoFrame class represents a frame of video data.

Details
A QVideoFrame encapsulates the pixel data of a video frame, and information about the frame.

Video frames can come from several places - decoded media , a camera , or generated programmatically. The way pixels are described in these frames can vary greatly, and some pixel formats offer greater compression opportunities at the expense of ease of use.

The pixel contents of a video frame can be mapped to memory using the map() function. After a successful call to map() , the video data can be accessed through various functions. Some of the YUV pixel formats provide the data in several planes. The planeCount() method will return the amount of planes that being used.

While mapped, the video data of each plane can accessed using the bits() function, which returns a pointer to a buffer. The size of this buffer is given by the mappedBytes() function, and the size of each line is given by bytesPerLine() . The return value of the handle() function may also be used to access frame data using the internal buffer’s native APIs (for example - an OpenGL texture handle).

A video frame can also have timestamp information associated with it. These timestamps can be used to determine when to start and stop displaying the frame.

QVideoFrame objects can consume a significant amount of memory or system resources and should not be held for longer than required by the application.

Note

Since video frames can be expensive to copy, QVideoFrame is explicitly shared, so any change made to a video frame will also apply to any copies.

See also

QAbstractVideoBuffer QVideoFrameFormat MapMode

Synopsis
Methods
def __init__()

def bytesPerLine()

def endTime()

def handleType()

def height()

def isMapped()

def isReadable()

def isValid()

def isWritable()

def map()

def mapMode()

def mappedBytes()

def mirrored()

def __ne__()

def __eq__()

def paint()

def pixelFormat()

def planeCount()

def rotation()

def rotationAngle()

def setEndTime()

def setMirrored()

def setRotation()

def setRotationAngle()

def setStartTime()

def setStreamFrameRate()

def setSubtitleText()

def size()

def startTime()

def streamFrameRate()

def subtitleText()

def surfaceFormat()

def swap()

def toImage()

def unmap()

def videoBuffer()

def width()

Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

class HandleType
Identifies the type of a video buffers handle.

Constant

Description

QVideoFrame.HandleType.NoHandle

The buffer has no handle, its data can only be accessed by mapping the buffer.

QVideoFrame.HandleType.RhiTextureHandle

The handle of the buffer is defined by The Qt Rendering Hardware Interface (RHI). RHI is Qt’s internal graphics abstraction for 3D APIs, such as OpenGL, Vulkan, Metal, and Direct 3D.

See also

handleType()

class MapMode
Enumerates how a video buffer’s data is mapped to system memory.

Constant

Description

QVideoFrame.MapMode.NotMapped

The video buffer is not mapped to memory.

QVideoFrame.MapMode.ReadOnly

The mapped memory is populated with data from the video buffer when mapped, but the content of the mapped memory may be discarded when unmapped.

QVideoFrame.MapMode.WriteOnly

The mapped memory is uninitialized when mapped, but the possibly modified content will be used to populate the video buffer when unmapped.

QVideoFrame.MapMode.ReadWrite

The mapped memory is populated with data from the video buffer, and the video buffer is repopulated with the content of the mapped memory when it is unmapped.

See also

mapMode() map()

Added in version 6.1.

class RotationAngle
Use Rotation instead.

The angle of the clockwise rotation that should be applied to a video frame before displaying.

Constant

Description

QVideoFrame.RotationAngle.Rotation0

No rotation required, the frame has correct orientation

QVideoFrame.RotationAngle.Rotation90

The frame should be rotated by 90 degrees

QVideoFrame.RotationAngle.Rotation180

The frame should be rotated by 180 degrees

QVideoFrame.RotationAngle.Rotation270

The frame should be rotated by 270 degrees

Added in version 6.2.3.

__init__()
Constructs a null video frame.

__init__(image)
Parameters:
image – QImage

Constructs a QVideoFrame from a QImage.

If the QImage::Format matches one of the formats in PixelFormat , the QVideoFrame will hold an instance of the image and use that format without any pixel format conversion. In this case, pixel data will be copied only if you call map with WriteOnly flag while keeping the original image.

Otherwise, if the QImage::Format matches none of video formats, the image is first converted to a supported (A)RGB format using QImage::convertedTo() with the Qt::AutoColor flag. This may incur a performance penalty.

If QImage::isNull() evaluates to true for the input QImage, the QVideoFrame will be invalid and isValid() will return false.

See also

pixelFormatFromImageFormat() isNull()

__init__(other)
Parameters:
other – QVideoFrame

Constructs a shallow copy of other. Since QVideoFrame is explicitly shared, these two instances will reflect the same frame.

__init__(format)
Parameters:
format – QVideoFrameFormat

Constructs a video frame of the given pixel format.

__init__(buffer, format)
Parameters:
buffer – QAbstractVideoBuffer

format – QVideoFrameFormat

Note

This function is deprecated.

bytesPerLine(plane)
Parameters:
plane – int

Return type:
int

Returns the number of bytes in a scan line of a plane.

This value is only valid while the frame data is mapped .

See also

bits() map() mappedBytes() planeCount()

endTime()
Return type:
int

Returns the presentation time (in microseconds) when a frame should stop being displayed.

An invalid time is represented as -1.

See also

setEndTime()

handleType()
Return type:
HandleType

Returns the type of a video frame’s handle.

The handle type could either be NoHandle , meaning that the frame is memory based, or a RHI texture.

height()
Return type:
int

Returns the height of a video frame.

isMapped()
Return type:
bool

Identifies if a video frame’s contents are currently mapped to system memory.

This is a convenience function which checks that the MapMode of the frame is not equal to NotMapped .

Returns true if the contents of the video frame are mapped to system memory, and false otherwise.

See also

mapMode() MapMode

isReadable()
Return type:
bool

Identifies if the mapped contents of a video frame were read from the frame when it was mapped.

This is a convenience function which checks if the MapMode contains the WriteOnly flag.

Returns true if the contents of the mapped memory were read from the video frame, and false otherwise.

See also

mapMode() MapMode

isValid()
Return type:
bool

Identifies whether a video frame is valid.

An invalid frame has no video buffer associated with it.

Returns true if the frame is valid, and false if it is not.

isWritable()
Return type:
bool

Identifies if the mapped contents of a video frame will be persisted when the frame is unmapped.

This is a convenience function which checks if the MapMode contains the WriteOnly flag.

Returns true if the video frame will be updated when unmapped, and false otherwise.

Note

The result of altering the data of a frame that is mapped in read-only mode is undefined. Depending on the buffer implementation the changes may be persisted, or worse alter a shared buffer.

See also

mapMode() MapMode

map(mode)
Parameters:
mode – MapMode

Return type:
bool

Maps the contents of a video frame to system (CPU addressable) memory.

In some cases the video frame data might be stored in video memory or otherwise inaccessible memory, so it is necessary to map a frame before accessing the pixel data. This may involve copying the contents around, so avoid mapping and unmapping unless required.

The map mode indicates whether the contents of the mapped memory should be read from and/or written to the frame. If the map mode includes the QVideoFrame::ReadOnly flag the mapped memory will be populated with the content of the video frame when initially mapped. If the map mode includes the QVideoFrame::WriteOnly flag the content of the possibly modified mapped memory will be written back to the frame when unmapped.

While mapped the contents of a video frame can be accessed directly through the pointer returned by the bits() function.

When access to the data is no longer needed, be sure to call the unmap() function to release the mapped memory and possibly update the video frame contents.

If the video frame has been mapped in read only mode, it is permissible to map it multiple times in read only mode (and unmap it a corresponding number of times). In all other cases it is necessary to unmap the frame first before mapping a second time.

Note

Writing to memory that is mapped as read-only is undefined, and may result in changes to shared data or crashes.

Returns true if the frame was mapped to memory in the given mode and false otherwise.

See also

unmap() mapMode() bits()

mapMode()
Return type:
MapMode

Returns the mode a video frame was mapped to system memory in.

See also

map() MapMode

mappedBytes(plane)
Parameters:
plane – int

Return type:
int

Returns the number of bytes occupied by plane plane of the mapped frame data.

This value is only valid while the frame data is mapped .

See also

map()

mirrored()
Return type:
bool

Returns whether the frame should be mirrored around its vertical axis before displaying.

Transformations of QVideoFrame, specifically rotation and mirroring, are used only for displaying the video frame and are applied on top of the surface transformation, which is determined by QVideoFrameFormat . Mirroring is applied after rotation.

Mirroring is typically needed for video frames coming from a front camera of a mobile device.

See also

setMirrored()

__ne__(other)
Parameters:
other – QVideoFrame

Return type:
bool

Returns true if this QVideoFrame and other do not reflect the same frame.

__eq__(other)
Parameters:
other – QVideoFrame

Return type:
bool

Returns true if this QVideoFrame and other reflect the same frame.

paint(painter, rect, options)
Parameters:
painter – QPainter

rect – QRectF

options – PaintOptions

Uses a QPainter, painter, to render this QVideoFrame to rect. The PaintOptions options can be used to specify a background color and how rect should be filled with the video.

Note

that rendering will usually happen without hardware acceleration when using this method.

pixelFormat()
Return type:
PixelFormat

Returns the pixel format of this video frame.

planeCount()
Return type:
int

Returns the number of planes in the video frame.

See also

map()

rotation()
Return type:
Rotation

Returns the angle the frame should be rotated clockwise before displaying.

Transformations of QVideoFrame, specifically rotation and mirroring, are used only for displaying the video frame and are applied on top of the surface transformation, which is determined by QVideoFrameFormat . Rotation is applied before mirroring.

See also

setRotation()

rotationAngle()
Return type:
RotationAngle

Note

This function is deprecated.

Use QVideoFrame::rotation instead.

Returns the angle the frame should be rotated clockwise before displaying.

See also

setRotationAngle()

setEndTime(time)
Parameters:
time – int

Sets the presentation time (in microseconds) when a frame should stop being displayed.

An invalid time is represented as -1.

See also

endTime()

setMirrored(mirrored)
Parameters:
mirrored – bool

Sets whether the frame should be mirrored around its vertical axis before displaying.

Transformations of QVideoFrame, specifically rotation and mirroring, are used only for displaying the video frame and are applied on top of the surface transformation, which is determined by QVideoFrameFormat . Mirroring is applied after rotation.

Mirroring is typically needed for video frames coming from a front camera of a mobile device.

Default value is false.

See also

mirrored()

setRotation(angle)
Parameters:
angle – Rotation

Sets the angle the frame should be rotated clockwise before displaying.

Transformations of QVideoFrame, specifically rotation and mirroring, are used only for displaying the video frame and are applied on top of the surface transformation, which is determined by QVideoFrameFormat . Rotation is applied before mirroring.

Default value is QtVideo::Rotation::None.

See also

rotation()

setRotationAngle(angle)
Parameters:
angle – RotationAngle

Note

This function is deprecated.

Use QVideoFrame::setRotation instead.

Sets the angle the frame should be rotated clockwise before displaying.

See also

rotationAngle()

setStartTime(time)
Parameters:
time – int

Sets the presentation time (in microseconds) when the frame should initially be displayed.

An invalid time is represented as -1.

See also

startTime()

setStreamFrameRate(rate)
Parameters:
rate – float

Sets the frame rate of a video stream in frames per second.

See also

streamFrameRate()

setSubtitleText(text)
Parameters:
text – str

Sets the subtitle text that should be rendered together with this video frame to text.

See also

subtitleText()

size()
Return type:
QSize

Returns the dimensions of a video frame.

startTime()
Return type:
int

Returns the presentation time (in microseconds) when the frame should be displayed.

An invalid time is represented as -1.

See also

setStartTime()

streamFrameRate()
Return type:
float

Returns the frame rate of a video stream in frames per second.

See also

setStreamFrameRate()

subtitleText()
Return type:
str

Returns the subtitle text that should be rendered together with this video frame.

See also

setSubtitleText()

surfaceFormat()
Return type:
QVideoFrameFormat

Returns the surface format of this video frame.

swap(other)
Parameters:
other – QVideoFrame

Swaps the current video frame with other.

toImage()
Return type:
QImage

Converts current video frame to image.

The conversion is based on the current pixel data and the surface format . Transformations of the frame don’t impact the result since they are applied for presentation only.

unmap()
Releases the memory mapped by the map() function.

If the MapMode included the WriteOnly flag this will persist the current content of the mapped memory to the video frame.

unmap() should not be called if map() function failed.

See also

map()

videoBuffer()
Return type:
QAbstractVideoBuffer

Note

This function is deprecated.

width()
Return type:
int

Returns the width of a video frame.

class PaintOptions
Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

class PaintFlag
PySide6.QtMultimedia.QVideoFrame.PaintOptions.backgroundColor
PySide6.QtMultimedia.QVideoFrame.PaintOptions.aspectRatioMode
PySide6.QtMultimedia.QVideoFrame.PaintOptions.paintFlags

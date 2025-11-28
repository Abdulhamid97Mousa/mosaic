PySide6.QtMultimedia.QVideoFrameInput
class QVideoFrameInput
The QVideoFrameInput class is used for providing custom video frames to QMediaRecorder or a video output through QMediaCaptureSession .

Details
Warning

This section contains snippets that were automatically translated from C++ to Python and may contain errors.

QVideoFrameInput is only supported with the FFmpeg backend.

Custom video frames can be recorded by connecting a QVideoFrameInput and a QMediaRecorder to a QMediaCaptureSession . For a pull mode implementation, call sendVideoFrame() in response to the readyToSendVideoFrame() signal. In the snippet below this is done by connecting the signal to a slot in a custom media generator class. The slot function emits another signal with a new video frame, which is connected to sendVideoFrame() :

session = QMediaCaptureSession()
recorder = QMediaRecorder()
videoInput = QVideoFrameInput()
session.setRecorder(recorder)
session.setVideoFrameInput(videoInput)
MediaGenerator generator # Custom class providing video frames
videoInput.readyToSendVideoFrame.connect(
        generator.nextVideoFrame)
generator.videoFrameReady.connect(
        videoInput.sendVideoFrame)
recorder.record()
Here’s a minimal implementation of the slot function that provides video frames:

def nextVideoFrame(self):

    frame = nextFrame()
    videoFrameReady.emit(frame)
For more details see readyToSendVideoFrame() and sendVideoFrame() .

See also

QMediaRecorder QMediaCaptureSession QVideoSink


Added in version 6.8.

Synopsis
Methods
def __init__()

def captureSession()

def format()

def sendVideoFrame()

Signals
def readyToSendVideoFrame()

Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

__init__([parent=None])
Parameters:
parent – QObject

Constructs a new QVideoFrameInput object with parent.

__init__(format[, parent=None])
Parameters:
format – QVideoFrameFormat

parent – QObject

Constructs a new QVideoFrameInput object with video frame format and parent.

The specified format will work as a hint for the initialization of the matching video encoder upon invoking record() . If the format is not specified or not valid, the video encoder will be initialized upon sending the first frame. Sending of video frames with another pixel format and size after initialization of the matching video encoder might cause a performance penalty during recording.

We recommend specifying the format if you know in advance what kind of frames you’re going to send.

captureSession()
Return type:
QMediaCaptureSession

Returns the capture session this video frame input is connected to, or a nullptr if the video frame input is not connected to a capture session.

Use setVideoFrameInput() to connect the video frame input to a session.

format()
Return type:
QVideoFrameFormat

Returns the video frame format that was specified upon construction of the video frame input.

readyToSendVideoFrame()
Signals that a new frame can be sent to the video frame input. After receiving the signal, if you have frames to be sent, invoke sendVideoFrame once or in a loop until it returns false.

See also

sendVideoFrame()

sendVideoFrame(frame)
Parameters:
frame – QVideoFrame

Return type:
bool

Sends QVideoFrame to QMediaRecorder or a video output through QMediaCaptureSession .

Returns true if the specified frame has been sent successfully to the destination. Returns false, if the frame hasn’t been sent, which can happen if the instance is not assigned to QMediaCaptureSession , the session doesn’t have video outputs or a media recorder, the media recorder is not started or its queue is full. The signal readyToSendVideoFrame will be sent as soon as the destination is able to handle a new frame.

Sending of an empty video frame is treated by QMediaRecorder as an end of the input stream. QMediaRecorder stops the recording automatically if autoStop is true and all the inputs have reported the end of the stream.
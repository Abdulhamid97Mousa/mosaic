Integrating with Qt
Basic organization
Valkka can be used with any GUI framework, say, with GTK or Qt. Here we have an emphasis on Qt, but the general guidelines discussed here, apply to any other GUI framework as well. Concrete examples are provided only for Qt.

At the GUI’s main window constructor:

Start your python multiprocesses if you have them (typically used for machine vision analysis)

Instantiate filtergraphs (from dedicated filtergraph classes, like we did in tutorial)

Start all libValkka threads (LiveThread, OpenGLThread, etc.)

Start a QThread listening to your python multiprocesses (1), in order to translate messages from multiprocesses to Qt signals.

Finally:

Start your GUI framework’s execution loop

At main window close event, close all threads, filterchains and multiprocesses

Examples of all this can be found in the PyQt testsuite together with several filtergraph classes.

Drawing video into a widget
X-windows, i.e. “widgets” in the Qt slang, can be created at the Qt side and passed to Valkka. Alternatively, x-windows can be created at the Valkka side and passed to Qt as “foreign widgets”.

As you learned in the tutorial, we use the X-window window ids like this:

context_id=glthread.newRenderContextCall(1,window_id,0)
That creates a mapping: all frames with slot number “1” are directed to an X-window with a window id “window_id” (the last number “0” is the z-stacking and is not currently used).

We can use the window id of an existing Qt widget “some_widget” like this:

window_id=int(some_widget.winId())
There is a stripped-down example of this in

valkka_examples/api_level_1/qt/

  single_stream_rtsp.py
You can also let Valkka create the X-window (with correct visual parameters, no XSignals, etc.) and embed that X-window into Qt. This can be done with:

foreign_window =QtGui.QWindow.fromWinId(win_id)
foreign_widget =QtWidgets.QWidget.createWindowContainer(foreign_window,parent=parent)
where “win_id” is the window_id returned by Valkka, “parent” is the parent widget of the widget we’re creating here and “foreign_widget” is the resulting widget we’re going to use in Qt.

However, “foreign_widget” created this way does not catch mouse gestures. This can be solved by placing a “dummy” QWidget on top of the “foreign_widget” (using a layout). An example of this can be found in

Qt with multiprocessing
Using python multiprocesses with Qt complicates things a bit: we need a way to map messages from the multiprocess into signals at the main Qt program. This can be done by communicating with the python multiprocess via pipes and converting the pipe messages into incoming and outgoing Qt signals.

Let’s state that graphically:

Qt main loop running with signals and slots
    |
    +--- QThread receiving/sending signals --- writing/reading communication pipes
                                                                     |
                                                       +-------------+------+----------------+
                                                       |                    |                |
                                                      multiprocess_1   multiprocess_2  multiprocess_3

                                                       python multiprocesses doing their thing
                                                       and writing/reading their communication pipes
                                                       ==> subclass from valkka.multiprocess.MessageProcess
Note that we only need a single QThread to control several multiprocesses.

We will employ the valkka-multiprocess module to couple Qt signals and slots with multiprocesses:

+--------------------------------------+
|                                      |
| QThread                              |
|  watching the communication pipe     |
|                   +----- reads "ping"|
|                   |               |  |
+-------------------|------------------+
                    |               |
 +------------------|-------+       |        ...
 | Frontend methods |       |       ^          :
 |                  |       |      pipe        :
 | def ping():  <---+       |       |          :
 |   do something           |       |          :
 |   (say, send a qt signal)|       |          :
 |                          |       |          :
 | def pong(): # qt slot    |       |          :
 |   sendSignal("pong") ---------+  |          :
 |                          |    |  |          :    valkka.multiprocess.MessageProcess
 +--------------------------+    |  |          :
 | Backend methods          |    |  |          :    Backend is running in the "background" in its own virtual memory space
 |                          |    |  |          :
 | sendSignal__("ping") ------->----+          :
 |                          |    |             :
 | watching childpipe <------- childpipe       :
 |                 |        |                  :
 | def pong__(): <-+        |                  :
 |  do something            |                  :
 |                          |                  :
 +--------------------------+                ..:
The class valkka.multiprocess.MessageProcess provides a model class that has been derived from python’s multiprocessing.Process class. In MessageProcess, the class has both “frontend” and “backend” methods.

The MessageProcess class comes with the main libValkka package, but you can also install it separately.

I recommend that you read that valkka-multiprocess documentation as it is important to understand what you are doing here - what is running in the “background” and what in your main python (Qt) process as including libValkka threads and QThreads into the same mix can easily result in the classical “fork-combined-with-threading” pitfall, leading to a leaky-crashy program.

Please refer also to the PyQt testsuite how to do things correctly.

A simplified, stand-alone python multiprocessing/Qt sample program is provided here (without any libValkka components):

valkka_examples/api_level_2/qt/

    multiprocessing_demo.py
Try it to see the magic of python multiprocessing connected with the Qt signal/slot system.

Finally, for creating a libValkka Qt application having a frontend QThread, that controls OpenCV process(es), take a look at

valkka_examples/api_level_2/qt/

    test_studio_detector.py
And follow the code therein. You will find these classes:

MovementDetectorProcess : multiprocess with Qt signals and OpenCV

QHandlerThread : the frontend QThread

C++ API
There is no obligation to use Valkka from python - the API is usable from cpp as well: all python libValkka threads and filters are just swig-wrapped cpp code.

If programming in Qt with C++ is your thing, then you can just forget all that multiprocessing considered here and use cpp threads instead.

Say, you can use Valkka’s FrameFifo and Thread infrastructure to create threads that read frames and feed them to an OpenCV analyzer (written in cpp).

You can also communicate from your custom cpp thread to the python side. A python program using an example cpp thread (TestThread) which communicates with PyQt signals and slots can be found here:

valkka_examples/api_level_2/qt/

    cpp_thread_demo.py



Integrating with Qt
Basic organization
Valkka can be used with any GUI framework, say, with GTK or Qt. Here we have an emphasis on Qt, but the general guidelines discussed here, apply to any other GUI framework as well. Concrete examples are provided only for Qt.

At the GUI’s main window constructor:

Start your python multiprocesses if you have them (typically used for machine vision analysis)

Instantiate filtergraphs (from dedicated filtergraph classes, like we did in tutorial)

Start all libValkka threads (LiveThread, OpenGLThread, etc.)

Start a QThread listening to your python multiprocesses (1), in order to translate messages from multiprocesses to Qt signals.

Finally:

Start your GUI framework’s execution loop

At main window close event, close all threads, filterchains and multiprocesses

Examples of all this can be found in the PyQt testsuite together with several filtergraph classes.

Drawing video into a widget
X-windows, i.e. “widgets” in the Qt slang, can be created at the Qt side and passed to Valkka. Alternatively, x-windows can be created at the Valkka side and passed to Qt as “foreign widgets”.

As you learned in the tutorial, we use the X-window window ids like this:

context_id=glthread.newRenderContextCall(1,window_id,0)
That creates a mapping: all frames with slot number “1” are directed to an X-window with a window id “window_id” (the last number “0” is the z-stacking and is not currently used).

We can use the window id of an existing Qt widget “some_widget” like this:

window_id=int(some_widget.winId())
There is a stripped-down example of this in

valkka_examples/api_level_1/qt/

  single_stream_rtsp.py
You can also let Valkka create the X-window (with correct visual parameters, no XSignals, etc.) and embed that X-window into Qt. This can be done with:

foreign_window =QtGui.QWindow.fromWinId(win_id)
foreign_widget =QtWidgets.QWidget.createWindowContainer(foreign_window,parent=parent)
where “win_id” is the window_id returned by Valkka, “parent” is the parent widget of the widget we’re creating here and “foreign_widget” is the resulting widget we’re going to use in Qt.

However, “foreign_widget” created this way does not catch mouse gestures. This can be solved by placing a “dummy” QWidget on top of the “foreign_widget” (using a layout). An example of this can be found in

Qt with multiprocessing
Using python multiprocesses with Qt complicates things a bit: we need a way to map messages from the multiprocess into signals at the main Qt program. This can be done by communicating with the python multiprocess via pipes and converting the pipe messages into incoming and outgoing Qt signals.

Let’s state that graphically:

Qt main loop running with signals and slots
    |
    +--- QThread receiving/sending signals --- writing/reading communication pipes
                                                                     |
                                                       +-------------+------+----------------+
                                                       |                    |                |
                                                      multiprocess_1   multiprocess_2  multiprocess_3

                                                       python multiprocesses doing their thing
                                                       and writing/reading their communication pipes
                                                       ==> subclass from valkka.multiprocess.MessageProcess
Note that we only need a single QThread to control several multiprocesses.

We will employ the valkka-multiprocess module to couple Qt signals and slots with multiprocesses:

+--------------------------------------+
|                                      |
| QThread                              |
|  watching the communication pipe     |
|                   +----- reads "ping"|
|                   |               |  |
+-------------------|------------------+
                    |               |
 +------------------|-------+       |        ...
 | Frontend methods |       |       ^          :
 |                  |       |      pipe        :
 | def ping():  <---+       |       |          :
 |   do something           |       |          :
 |   (say, send a qt signal)|       |          :
 |                          |       |          :
 | def pong(): # qt slot    |       |          :
 |   sendSignal("pong") ---------+  |          :
 |                          |    |  |          :    valkka.multiprocess.MessageProcess
 +--------------------------+    |  |          :
 | Backend methods          |    |  |          :    Backend is running in the "background" in its own virtual memory space
 |                          |    |  |          :
 | sendSignal__("ping") ------->----+          :
 |                          |    |             :
 | watching childpipe <------- childpipe       :
 |                 |        |                  :
 | def pong__(): <-+        |                  :
 |  do something            |                  :
 |                          |                  :
 +--------------------------+                ..:
The class valkka.multiprocess.MessageProcess provides a model class that has been derived from python’s multiprocessing.Process class. In MessageProcess, the class has both “frontend” and “backend” methods.

The MessageProcess class comes with the main libValkka package, but you can also install it separately.

I recommend that you read that valkka-multiprocess documentation as it is important to understand what you are doing here - what is running in the “background” and what in your main python (Qt) process as including libValkka threads and QThreads into the same mix can easily result in the classical “fork-combined-with-threading” pitfall, leading to a leaky-crashy program.

Please refer also to the PyQt testsuite how to do things correctly.

A simplified, stand-alone python multiprocessing/Qt sample program is provided here (without any libValkka components):

valkka_examples/api_level_2/qt/

    multiprocessing_demo.py
Try it to see the magic of python multiprocessing connected with the Qt signal/slot system.

Finally, for creating a libValkka Qt application having a frontend QThread, that controls OpenCV process(es), take a look at

valkka_examples/api_level_2/qt/

    test_studio_detector.py
And follow the code therein. You will find these classes:

MovementDetectorProcess : multiprocess with Qt signals and OpenCV

QHandlerThread : the frontend QThread

C++ API
There is no obligation to use Valkka from python - the API is usable from cpp as well: all python libValkka threads and filters are just swig-wrapped cpp code.

If programming in Qt with C++ is your thing, then you can just forget all that multiprocessing considered here and use cpp threads instead.

Say, you can use Valkka’s FrameFifo and Thread infrastructure to create threads that read frames and feed them to an OpenCV analyzer (written in cpp).

You can also communicate from your custom cpp thread to the python side. A python program using an example cpp thread (TestThread) which communicates with PyQt signals and slots can be found here:

valkka_examples/api_level_2/qt/

    cpp_thread_demo.py
See also the documentation for the cpp source code of TestThread



Installing
The debian package includes the core library, its python bindings and some API level 2 python code. The python part is installed “globally” into /usr/lib/python3/dist-packages/

Note

LibValkka comes precompiled and packaged for a certain ubuntu distribution version. This means that the compilation and it’s dependencies assume the default python version of that distribution. Using custom-installed python versions, anacondas and whatnot might cause dependency problems.

A. Install using PPA
the preferred way

For recent ubuntu distributions, the core library binary packages and python bindings are provided by a PPA repository. Subscribe to the PPA repo (do this only once) with:

sudo apt-add-repository ppa:sampsa-riikonen/valkka
Install with:

sudo apt-get update
sudo apt-get install valkka
When you need to update valkka, do:

sudo apt-get update
sudo apt-get install --only-upgrade valkka
B. Install using releases
if you don’t like PPAs

You can download and install the required .deb packages “manually” from the releases page

sudo dpkg -i Valkka-*.deb
sudo apt-get install -fy
The last line pulls the dependencies.

Repeat the process when you need to update.

C. Compile yourself
the last resort

If you’re not using a recent Ubuntu distro and need to build libValkka and it’s python bindings yourself, please refer to the valkka-core github page.

Test your installation
Test the installation with:

curl https://raw.githubusercontent.com/elsampsa/valkka-examples/master/quicktest.py -o quicktest.py
python3 quicktest.py
Numpy
Valkka-core binaries has been compiled with the numpy version that comes with the corresponding Ubuntu distro, i.e. the numpy you would install with sudo apt-get install python3-numpy.

That version is automatically installed when you install valkka core with sudo apt-get, but it might be “shadowed” by your locally installed numpy.

If you get errors about numpy import, try removing your locally installed numpy (i.e. the version you installed with pip install --user).

Install the testsuite
First, install some debian packages:

sudo apt-get install python3-pip git mesa-utils ffmpeg vlc
some of these will be used for benchmarking Valkka agains other programs.

The testsuite and tutorials use also imutils and PyQt5, so install a fresh version of them locally with pip:

pip3 install --user imutils PyQt5 PySide2 setproctitle
Here we have installed two flavors of the Qt python bindings, namely, PyQt5 and PySide2. They can be used in an identical manner. If you use PyQt5, be aware of its licensing terms.

Finally, for tutorial code and the PyQt test suite, download valkka-examples with:

git clone https://github.com/elsampsa/valkka-examples
Test the installation with:

cd valkka-examples
python3 quicktest.py
and you’re all set.

When updating the python examples (do this always after updating valkka-core), do the following:

git pull
python3 quicktest.py
This checks that valkka-core and valkka-examples have consistent versions.

In the case of a numerical python version mismatch error, you are not using the default numpy provided by your Ubuntu distribution (from the debian package python3-numpy). Remove the conflicting numpy installation with pip3 uninstall or setting up a virtualenv.

Next, try out the PyQt test/demo suite or learn to program with the tutorial.

GTK
If you wan’t to use GTK as your graphical user interface, you must install the PyGObject python bindings, as instructed here, namely:

sudo apt-get install python-gi python-gi-cairo python3-gi python3-gi-cairo gir1.2-gtk-3.0
OpenCV
Install with:

pip3 uninstall opencv-python
sudo pip3 uninstall opencv-python # just in case!
sudo apt-get install python3-opencv
The first one deinstall anything you may have installed with pip, while the second one installs the (good) opencv that comes with your linux distro’s default python opencv installation.

Development version
As described above, for the current stable version of valkka-core, just use the repository.

For the development version (with experimental and unstable features) you have to compile from source. You might need to do this also for architectures other than x86.

Decoding
Single thread
By default, libValkka uses only a single core per decoder (the decoding threads can also be bound to a certain core - see the testsuite for more details).

This is a good idea if you have a large number of light streams. What is exactly a light stream depends on your linux box, but let’s assume here that it is a 1080p video running approx. at 20 frames per second.

Multithread
If you need to use a single heavy stream, then you might want to dedicate several cores in decoding a single stream. A heavy stream could be example that 4000x3000 4K camera of yours running at 60 frames per second (!)

However, before using such a beast, you must ask yourself, do you really need something like that?

The biggest screen you’ll ever be viewing the video from, is probably 1080p, while a framerate of 15 fps is good for the human eye. Modern convoluted neural networks (yolo, for example), are using typically a resolution of ~ 600x600 pixels and analyzing max. 30 frames per seconds. And we still haven’t talked about clogging up your LAN.

If you really, really have to use several threads per decoder, modify tutorial’s lesson 2 like this:

avthread = AVThread("avthread",info_filter)
avthread.setNumberOfThreads(4)
That will dedicate four cores to the decoder. Remember to call setNumberOfThreads before starting the AVThread instance.

GPU Accelerated
Hardware (hw) accelerated decoders are available in libValkka. For more details, please see here. However, before using them, you should ask yourself if you really need them. Maybe it is better to save all GPU muscle for deep learning inference instead?

Video hw acceleration libraries are typically closed-source implementations, and the underlying “black-box” can be poorly implemented and suffer from memory leaks. Read for example this thread. Slowly accumulating memleaks are poison for live video streaming applications which are supposed to stream continuously for days, weeks and even forever.

Sometimes the proprietary libraries may also restrict how many simultaneous hw video decoders you can have, while there are no such restrictions on CPU decoding.

So, if you have a linux box, dedicated solely for streaming and with decent CPU(s), don’t be over-obsessed with hw decoding.

Queueing frames
Typically, when decoding H264 video, handling the intra-frame takes much more time than decoding the consecutive B- and P-frames. This is very pronounced for heavy streams (see above).

Because of that the intra frame will arrive late for the presentation, while the consecutive frames arrive in a burst.

This problem can be solved with buffering. Modify tutorial’s lesson 3 like this:

from valkka.core import *

glthread = OpenGLThread ("glthread")

gl_ctx = core.OpenGLFrameFifoContext()
gl_ctx.n_720p = 0
gl_ctx.n_1080p = 0
gl_ctx.n_1440p = 0
gl_ctx.n_4K = 40

glthread = OpenGLThread("glthread", gl_ctx, 500)
That will reserve 40 4K frames for queueing and presentation of video, while the buffering time is 500 milliseconds.

For level 2 API, it would look like this:

from valkka.api2 import *

glthread = OpenGLThread(
    name ="glthread",
    n_720p = 0,
    n_1080p = 0,
    n_1440p = 0,
    n_4K = 40,
    msbuftime = 500
  )
Remember also that for certain class of frames (720p, 1080p, etc.):

number of pre-reserved frames >= total framerate x buffering time
For testing, you should use the test_studio_1.py program. See also this lesson of the tutorial.

Buffering solves many other issues as well. If you don’t get any image and the terminal screaming that “there are no more frames”, then just enhance the buffering.



The PyQt testsuite
So, you have installed valkka-core and valkka-examples as instructed here. The same hardware requirements apply here as in the tutorial.

The PyQt testsuite is available at

valkka_examples/api_level_2/qt/
The testsuite is intended for:

Demonstration

Benchmarking

Ultimate debugging

As materia prima for developers - take a copy of your own and use it as a basis for your own Valkka / Qt program

If you want a more serious demonstration, try out Valkka Live instead.

Currently the testsuite consists of the following programs:

File

Explanation

test_studio_1.py

- Stream from several rtsp cameras or sdp sources
- Widgets are grouped together
- This is just live streaming, so use:

rtsp://username:password@your_ip

- If you define a filename, it is interpreted as an sdp file

test_studio_2.py

- Like test_studio_1.py
- Floating widgets

test_studio_3.py

- Like test_studio_2.py
- On a multi-gpu system, video can be sent to another gpu/x-screen pair

test_studio_4.py

- Like test_studio_3.py
- A simple user menu where video widgets can be opened
- Open the File menu to add video on the screen

test_studio_detector.py

- Like test_studio_1.py
- Shares video to OpenCV processes
- OpenCV processes connect to Qt signal/slot system

test_studio_file.py

- Read and play stream from a matroska file
- Only matroska-contained h264 is accepted.
- Convert your video to “ip-camera-like” stream with:

ffmpeg -i your_video_file -c:v h264 -r 10 -preset ultrafast
-profile:v baseline
-bsf h264_mp4toannexb -x264-params keyint=10:min-keyint=10
-an outfile.mkv

- If you’re recording directly from an IP camera, use:

ffmpeg -i rtsp://username:password@your_ip -c:v copy -an outfile.mkv

test_studio_multicast.py

- Like test_studio_1.py
- Recast multiple IP cameras into multicast

test_studio_rtsp.py

- Like test_studio_1.py
- Recast IP cameras to unicast.
- Streams are accessible from local server:

rtsp://127.0.0.1:8554/streamN

(where N is the number of the camera)

test_studio_5.py

Very experimental
- Prefer test_studio_6.py over this one
- Continuous recording to ValkkaFS
- Dumps all streams to a single file (uses ValkkaMultiFS)
- Simultaneous, interactive playback : use mouse clicks & wheel to navigate the timeline
- remove directory fs_directory/ if the program refuses to start

test_studio_6.py

Experimental
- Continuous recording to ValkkaFS
- One file per stream (uses ValkkaSingleFS)
- Simultaneous, interactive playback : use mouse clicks & wheel to navigate the timeline
- remove directory fs_directory_*/ if the program refuses to start
Before launching any of the testsuite programs you should be aware of the common problems of linux video streaming.

test_studio_1.py
Do this:

cd valkka_examples/api_level_2/qt/
python3 test_studio_1.py
The program launches with the following menu:

_images/test_config.png
The field on the left is used to specify stream sources, one source per line. For IP cameras, use “rtsp://”, for sdp files, just give the filename. In the above example, we are connecting to two rtsp IP cams.

The fields on the right are:

Field name

What it does

n720p

Number of pre-reserved frames for 720p resolution

n1080p

Number of pre-reserved frames for 1080p resolution

n1440p

etc.

n4K

etc.

naudio

(not used)

verbose

(not used)

msbuftime

Frame buffering time in milliseconds

live affinity

Bind the streaming thread to a core. Default = -1 (no binding)

gl affinity

Bind frame presentation thread to a core. Default = -1

dec affinity start

Bind decoding threads to a core (first core). Default = -1

dec affinity stop

Bind decoding threads to cores (last core). Default = -1

replicate

Dump each stream to screen this many times

correct timestamp

1 = smart-correct timestamp (use this!)
0 = restamp upon arrival
socket size bytes

don’t touch. Default value = 0.

ordering time millisecs

don’t touch. Default value = 0.

As you learned from the tutorial, in Valkka, frames are pre-reserved on the GPU. If you’re planning to use 720p and 1080p cameras, reserve, say 200 frames for both.

Decoded frames are being queued for “msbuftime” milliseconds. This is necessary for de-jitter (among other things). The bigger the buffering time, the more pre-reserved frames you’ll need and the more lag you get into your live streaming. A nice value is 300. For more on the subject, read this.

Replicate demonstrates how Valkka can dump the stream (that’s decoded only once) to multiple X windows. Try for example the value 24 - you get each stream on the screen 24 times, without any performance degradation or the need to decode streams more than once.

In Valkka, all threads can be bound to a certain processor core. Default value “-1” indicates that the thread is unbound and that the kernel can switch it from one core to another (normal behaviour).

Let’s consider an example:

Field name

value

live affinity

0

gl affinity

1

dec affinity start

2

dec affinity stop

4

Now LiveThread (the thread that streams from cameras) stays at core index 0, all OpenGL operations and frame presenting at core index 1. Let’s imagine you have ten decoders running, then they will placed like this:

Core

Decoder thread

core 2

1, 4, 7, 10

core 3

2, 5, 8

core 4

3, 6, 9

Setting processor affinities might help, if you can afford the luxury of having one processor per decoder. Otherwise, it might mess up the load-balancing performed by the kernel.

By default, don’t touch the affinities (simply use the default value -1).

Finally, the buttons that launch the test, do the following:

Button

What it does?

SAVE

Saves the test configuration (yes, save it)

RUN(QT)

Runs THE TEST (after saving, press this!)

RUN

Runs the test without Qt

FFPLAY

Runs the streams in ffplay instead (if installed)

VLC

Runs the streams in vlc instead (if installed)

RUN(QT) is the thing you want to do.

FFPLAY and VLC launch the same rtsp streams by using either ffplay or vlc. This is a nice test to see how Valkka performs against some popular video players.

test_studio_detector.py
The detector test program uses OpenCV, so you need to have it installed

Launch the program like this:

cd valkka_examples/api_level_2/qt/
python3 test_studio_detector.py
This is similar to test_studio_1.py. In addition to presenting the streams on-screen, the decoded frames are passed, once in a second, to OpenCV movement detectors. When movement is detected, a signal is sent with the Qt signal/slot system to the screen.

This test program is also used in the gold standard test. Everything is here: streaming, decoding, OpenGL streaming, interface to python and even the posix shared memory and semaphores. One should be able to run this test with a large number of cameras for a long period of time without excessive memory consumption or system instabilities.
[![Logo](_static/logo.png)](index.html)

1.6.1

* [valkka-core](https://github.com/elsampsa/valkka-core)
* [valkka-examples](https://github.com/elsampsa/valkka-examples)
* [valkka-onvif](https://github.com/elsampsa/valkka-onvif)
* [valkka-multiprocess](https://elsampsa.github.io/valkka-multiprocess/_build/html/index.html)
* [valkka-streamer](https://github.com/elsampsa/valkka-streamer)
* [Issue Tracker](https://github.com/elsampsa/valkka-core/issues)
* [Package Repository](https://launchpad.net/~sampsa-riikonen/%2Barchive/ubuntu/valkka/%2Bpackages)
* [Dockerhub](https://hub.docker.com/r/elsampsa/valkka)
* [Valkka Live](https://elsampsa.github.io/valkka-live/)
* [Discord](https://discord.com/channels/1116100310574837812/1116101208797609994)

* [About Valkka](intro.html)
* [Supported hardware](hardware.html)
* [Installing](requirements.html)
* [The PyQt testsuite](testsuite.html)
* [Tutorial](tutorial.html)
* [Decoding](decoding.html)
* Integrating with Qt
  + [Basic organization](#basic-organization)
  + [Drawing video into a widget](#drawing-video-into-a-widget)
  + [Qt with multiprocessing](#qt-with-multiprocessing)
  + [C++ API](#c-api)
* [Multi-GPU systems](multi_gpu.html)
* [ValkkaFS](valkkafs.html)
* [Cloud Streaming](cloud.html)
* [OnVif & Discovery](onvif.html)
* [Common problems](pitfalls.html)
* [Debugging](debugging.html)
* [Repository Index](repos.html)
* [Licence & Copyright](license.html)
* [Authors](authors.html)
* [Knowledge Base](knowledge.html)

[Python Media Streaming Framework for Linux](index.html)

* Integrating with Qt
* [View page source](_sources/qt_notes.rst.txt)

---

# Integrating with Qt

## Basic organization

Valkka can be used with any GUI framework, say, with GTK or Qt. Here we have an emphasis on Qt, but the general guidelines discussed here, apply to any other GUI framework as well.
Concrete examples are provided only for Qt.

At the GUI’s main window constructor:

1. Start your python multiprocesses if you have them (typically used for machine vision analysis)
2. Instantiate filtergraphs (from dedicated filtergraph classes, like we did in [tutorial](lesson_3.html#multiple-streams))
3. Start all libValkka threads (LiveThread, OpenGLThread, etc.)
4. Start a QThread listening to your python multiprocesses (1), in order to translate messages from multiprocesses to Qt signals.

Finally:

5. Start your GUI framework’s execution loop
6. At main window close event, close all threads, filterchains and multiprocesses

Examples of all this can be found in [the PyQt testsuite](testsuite.html#testsuite) together with several filtergraph classes.

## Drawing video into a widget

X-windows, i.e. “widgets” in the Qt slang, can be created at the Qt side and passed to Valkka. Alternatively, x-windows can be created at the Valkka side and passed to Qt as “foreign widgets”.

As you learned in the tutorial, we use the X-window window ids like this:

```
context_id=glthread.newRenderContextCall(1,window_id,0)
```

That creates a mapping: all frames with slot number “1” are directed to an X-window with a window id “window_id” (the last number “0” is the z-stacking and is not currently used).

We can use the window id of an existing Qt widget “some_widget” like this:

```
window_id=int(some_widget.winId())
```

There is a stripped-down example of this in

```
valkka_examples/api_level_1/qt/

  single_stream_rtsp.py
```

You can also let Valkka create the X-window (with correct visual parameters, no XSignals, etc.) and embed that X-window into Qt. This can be done with:

```
foreign_window =QtGui.QWindow.fromWinId(win_id)
foreign_widget =QtWidgets.QWidget.createWindowContainer(foreign_window,parent=parent)
```

where “win_id” is the window_id returned by Valkka, “parent” is the parent widget of the widget we’re creating here and “foreign_widget” is the resulting widget we’re going to use in Qt.

However, “foreign_widget” created this way does not catch mouse gestures. This can be solved by placing a “dummy” QWidget on top of the “foreign_widget” (using a layout).
An example of this can be found in

## Qt with multiprocessing

Using python multiprocesses with Qt complicates things a bit: we need a way to map messages from the multiprocess into signals at the main Qt program.
This can be done by communicating with the python multiprocess via pipes and converting the pipe messages into incoming and outgoing Qt signals.

Let’s state that graphically:

```
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
```

Note that we only need a single QThread to control several multiprocesses.

We will employ the [valkka-multiprocess](https://elsampsa.github.io/valkka-multiprocess/_build/html/index.html) module
 to couple Qt signals and slots with multiprocesses:

```
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
```

The class **valkka.multiprocess.MessageProcess** provides a model class that has been derived from python’s **multiprocessing.Process** class.
In MessageProcess, the class has both “frontend” and “backend” methods.

The `MessageProcess` class comes with the main libValkka package, but you can also install it separately.

I recommend that you read that valkka-multiprocess documentation as it is important to understand what you are doing here - what is running in the “background”
 and what in your main python (Qt) process as including libValkka threads and QThreads into the same mix can easily result in the classical
 “fork-combined-with-threading” pitfall, leading to a leaky-crashy program.

Please refer also to [the PyQt testsuite](testsuite.html#testsuite) how to do things correctly.

A simplified, stand-alone python multiprocessing/Qt sample program is provided here (without any libValkka components):

```
valkka_examples/api_level_2/qt/

    multiprocessing_demo.py
```

Try it to see the magic of python multiprocessing connected with the Qt signal/slot system.

Finally, for creating a libValkka Qt application having a frontend QThread, that controls OpenCV process(es), take a look at

```
valkka_examples/api_level_2/qt/

    test_studio_detector.py
```

And follow the code therein. You will find these classes:

* *MovementDetectorProcess* : multiprocess with Qt signals and OpenCV
* *QHandlerThread* : the frontend QThread

## C++ API

There is no obligation to use Valkka from python - the API is usable from cpp as well: all python libValkka threads and filters are just swig-wrapped cpp code.

If programming in Qt with C++ is your thing, then you can just forget all that multiprocessing considered here and use cpp threads instead.

Say, you can use Valkka’s FrameFifo and Thread infrastructure to create threads that read frames and feed them to an OpenCV analyzer (written in cpp).

You can also communicate from your custom cpp thread to the python side. A python program using an example cpp thread (*TestThread*) which communicates with PyQt signals and slots can be found here:

```
valkka_examples/api_level_2/qt/

    cpp_thread_demo.py
```

See also the documentation for the cpp source code of [TestThread](https://elsampsa.github.io/valkka-core/html/classTestThread.html)

[Previous](decoding.html "Decoding")
[Next](multi_gpu.html "Multi-GPU systems")

---

© Copyright 2017-2020 Sampsa Riikonen.

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).

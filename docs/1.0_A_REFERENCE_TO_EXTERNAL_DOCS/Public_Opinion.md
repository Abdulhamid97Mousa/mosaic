Overcoming GUI Freezes in PyQt: From Threading & Multiprocessing to ZeroMQ & QProcess | by Foong Min Wong |


When working on a PyQt application, I encountered a graphical user interface (GUI) freezing issue while running a piece of computationally intensive code. PyQt applications use a main event loop to handle user interactions and GUI updates, since the code takes too long, it blocks the loop, causing the app to freeze.

I tried offloading the computation-heavy tasks to a separate thread. The computation is still freezing the GUI despite being in a thread. This might be due to insufficient resource contention. Also, the Global Interpreter Lock (GIL) in Python can bottleneck threads when performing CPU-intensive tasks. Then, I offloaded the computation to a separate process via multiprocessing instead of a thread, but I got an error stating that `cannot pickle ‘XXX’ object` , which contains a C# object that cannot be serialized directly in Python, so it cannot be used as an input argument. I found a way to overcome this by saving the program object as a temporary file and passing the temp file path as an argument. After resolving that small pickling error, I ran into another error: `QWidget: Must construct a QApplication before a QWidget`.

I stopped solving it for a week and have been thinking about how Jupyter Notebook enables users to run code interactively, and even when running an intensive operation such as plotting a graph with millions of data points or retrieving large amounts of data from the database, users can still create new and work on other Jupyter notebooks while waiting for those long computations to complete. We want to achieve something similar to Jupyter Notebook’s asynchronous communication between the front and backend, allowing the frontend to remain responsive while the backend processes long-running tasks. To achieve that, Jupyter uses `ZeroMQ`.

Then, I wondered if ZeroMQ could solve our GUI freezing issue with ZeroMQ handling communication and memory-intensive processes while PyQt provides the GUI. I reattempted to solve it via `[pyzmq](https://pypi.org/project/pyzmq/)` and … it works! By using ZeroMQ to communicate between the PyQt GUI and a separate plotting process, it isolates the memory consumption of the plotting operation. This prevents excessive memory usage in the GUI thread from impacting the responsiveness of the application.

The implementation is first by creating a separate worker process to listen on a ZeroMQ socket for incoming plotting requests from the GUI. Upon receiving the request, the worker processes, serializes plot data, and sends the serialized plot data back to the GUI/ main process. In the PyQt application, there will be a ZeroMQ socket to connect to the plotting process, send requests to the worker process, and receive the results such as serialized plot data. Finally, you can deserialize the plot data and update the plot in the PyQt GUI.

Although this approach works, I believe introducing ZeroMQ to our PyQt app might be overkill as it adds potential overhead from message passing, and complexity to the application. Port management also needs to be taken care of when using ZeroMQ, especially in the production environment in which hardcoded port numbers can lead to the risk of conflicts, nevertheless, we can mitigate this in several ways, for example by doing dynamic port allocation.

Example: [PyQt Plotting App via ZeroMQ](https://github.com/foongminwong/scripts/tree/main/pyqtgraph-zmq)

Press enter or click to view image in full size

Example: Plotting live simulated data via PyQtGraph

Given the potential complexity of ZeroMQ for this specific use case, `[QProcess](https://www.pythonguis.com/tutorials/qprocess-external-programs/#:~:text=like%20this%20%2D%2D-,QProcess%3A%20Destroyed%20while%20process%20%28%22python3%22%29%20is%20still,checking%20the%20value%20of%20self.)` might be a more suitable approach for offloading plotting tasks in our PyQt application. `QProcess` is part of Qt and is specifically designed for managing and communicating with external processes, making it well-suited for tasks like offloading CPU-bound operations. However, I got an error `QProcess: Destroyed while process is still running` when switching to `QProcess`. This will be my next experiment...



Wayland and Qt[¶](#wayland-and-qt "Link to this heading")
=========================================================

An overview of the Wayland protocol and how it fits into Qt.

Wayland was developed as an alternative to X11 on Linux. Its main purpose is to manage how the content of applications is displayed together on a shared screen, and how a user can interact with multiple applications sharing the same input devices.

This role in an operating system is often called a _display server_. The Wayland display server may also sometimes be called a _compositor_ and a _window manager_, referring to specific tasks it performs as part of its duty.

In the following, we will give a short introduction to Wayland and its role in Qt. For more details and background about Wayland itself, refer to the [official documentation](https://wayland.freedesktop.org/docs/html/) .

What is a Display Server[¶](#what-is-a-display-server "Link to this heading")
-----------------------------------------------------------------------------

The display server is the part of the operating system which manages the screen real estate and other shared resources. On a typical desktop system, you may have many independent applications running at the same time, each expecting to be able to render graphics to the screen and receive input.

The display server is a link between the application and shared resources such as a screen and input devices. A typical display server on a desktop system will place application content into distinct rectangular “windows”, which can be moved and resized by the user. The display server makes sure that the application content is displayed at the right position on the screen, that the active window receives input from the keyboard, that overlapping windows are drawn in the right order, and so on.

On other types of systems, the display server may be more restrictive. If the screen is an instrument panel in a car, or the control panel of a fork lift, for instance, then moving and resizing windows may not be desirable. Instead, each application may be locked into a pre-defined area of the screen and receive input from preassigned devices.

Either way, as long as there are multiple isolated processes competing for the same resources, a display server is useful.

The Role of Wayland[¶](#the-role-of-wayland "Link to this heading")
-------------------------------------------------------------------

The [Wayland](https://wayland.freedesktop.org/) name may refer to several related items:

> *   A set of protocols for communicating between a display server and its clients.
>     
> *   A library written in C with functions for inter-process communication, serving as the foundation for implementing said protocols.
>     
> *   An XML-based language for extending the protocol, as well as a tool for generating binding code in C from such extensions.
>     

Qt provides implementations for both the client and the server side of the protocol.

Normal Qt applications can be run as clients on a Wayland display server by selecting the “wayland” QPA plugin (this is the default on certain systems). You can do this by using the `-qpa wayland` option. In addition, the Qt Wayland Compositor module can be used to develop the display server itself.

Qt also has convenience functionality for easily extending the Wayland protocols with new interfaces.

Wayland and Other Technology[¶](#wayland-and-other-technology "Link to this heading")
-------------------------------------------------------------------------------------

On the Linux desktop, Wayland is an alternative to X11 and related extensions. It is a compositing display server at its core, and the term “compositor” is often used to describe the Wayland server. This means that clients will render content into an off-screen buffer, which will later be “composited” with other clients on the screen, allowing window effects such as drop shadows, transparency, background blurring, and so on.

One important design principle of the original X11 protocols is that the display server can be running on a thin terminal with only a screen and input devices. Its clients would then be running on remote systems with more processing power, communicating with the server over a network connection.

In contrast, Wayland is designed around the observation that, in modern setups, the client and display server are usually running on the same hardware. Distributed computing, remote storage and remote desktop functionality are usually handled through other mechanisms. Designing this into the protocol enables sharing graphics memory between the client and the server: When the compositor is placing client content on screen, it can simply copy it from one part of graphics memory to another.

For this to work optimally, the graphics driver must support Wayland. This support is provided through an extension to `EGL` which is called `EXT_platform_wayland`.

Note

Qt Wayland also supports compositing on systems where `EXT_platform_wayland` is not supported, either through `XComposite` or by copying application content to shared CPU memory. But for optimal performance, we recommend systems with driver support.

X11 has been extended to support features such as composition and direct rendering, but Wayland is designed around this use case from the ground up. It also aims to be small and extensible, in contrast to the complexity that has developed in X11 over time.

Extensibility and Embedded Systems[¶](#extensibility-and-embedded-systems "Link to this heading")
-------------------------------------------------------------------------------------------------

Since Wayland has a minimal core and is easily extensible, it is an ideal tool when building embedded Linux platforms.

Desktop-style window system features, for instance, are not part of the core protocol. Instead, Wayland has a special category of protocol extensions called “shells” that provide a way for the client to manage its surfaces. Desktop-style features are provided through a shell called `XDG Shell`. For other types of systems, a more specialized (and perhaps more restrictive) “shell” can be used. For instance, when making In-Vehicle Infotainment systems, the `IVI Shell` might be preferable.

The Wayland server broadcasts a list of its supported protocols (or “interfaces”) when a client connects, and the client can bind to the ones it wants to use. This can be any of the standard interfaces, but new extensions are also easy to add. Wayland defines an easily understandable XML format for defining protocols and the `waylandscanner` tool can be used to generate C code from these. (In Qt, we also have `qtwaylandscanner` which generates additional C++ binding code.)

After a client binds to an interface, it can make “requests” to the server and the server can send “events” to the client. The requests and events, as well as their arguments, are defined in the XML file describing the protocol.

For building a platform from scratch, when you control the code of both server and clients, adding extensions is an easy and controlled way of adding operating system features.

Multi-Process or Single-Process[¶](#multi-process-or-single-process "Link to this heading")
-------------------------------------------------------------------------------------------

When building a simple embedded platform with Qt, a perfectly viable option is to have all parts of the UI running in a single process. However, as the system becomes more complex, you may want to consider a multi-process system instead. This is where Wayland comes in. With Qt, at any point in your development process, you can choose to switch between single-process and multi-process.

Benefits of Multi-Process[¶](#benefits-of-multi-process "Link to this heading")
-------------------------------------------------------------------------------

The following diagrams illustrate the difference between multi-process and single-process systems.

> ![../_images/wayland-multi-process.png](../_images/wayland-multi-process.png)

Multi-Process Client Architecture

> ![../_images/wayland-single-process-eglfs.png](../_images/wayland-single-process-eglfs.png)

Single Process Client Architecture

The Qt Wayland Compositor module is ideal for creating the display server and compositor in multi-process systems on embedded Linux. The use of multi-process has the following benefits:

> *   [Stability](#wayland-and-qt)
>     
> *   [Security](#wayland-and-qt)
>     
> *   [Performance](#wayland-and-qt)
>     
> *   [Interoperability](#wayland-and-qt)
>     
> 
> Stability
> 
> Easier to recover when clients hang or crash
> 
> If you have a complex UI, then multi-process is useful because if one part of the UI crashes, it doesn’t affect the entire system. Similarly, the display won’t freeze, even when one client freezes.
> 
> Note
> 
> If your client is mandated by law to render safety-critical information, consider using [Qt Safe Renderer Overview](https://doc.qt.io/QtSafeRenderer/qtsr-overview.html) .
> 
> Protection against possible memory leaks
> 
> In a multi-process system, if one client has a memory leak and consumes lots of memory, that memory is recovered when that client exits. In contrast with single-process, the memory leak remains until the entire system restarts.
> 
> Security
> 
> In a single-process system, all clients can access each other’s memory. For example, there’s no isolation for sensitive data transfer; every line of code must be equally trustworthy. This isolation is there, by design, in multi-process systems.
> 
> Performance
> 
> If you have a CPU with multiple cores, a multi-process system can help distribute the load evenly across different cores, making more efficient use of your CPU.
> 
> Interoperability
> 
> You can interface with non-Qt clients in a multi-process system, as long as your clients understand Wayland or X11. For example, if you use gstreamer for video or if you want to use a navigation application built with another UI toolkit, you can run these clients alongside your other Qt-based clients.

Trade-offs of Multi-Process[¶](#trade-offs-of-multi-process "Link to this heading")
-----------------------------------------------------------------------------------

When going from single-process to multi-process, it is important to be conscious of the following trade-offs:

> *   [Increased video memory consumption](#wayland-and-qt)
>     
> *   [Increased main memory consumption](#wayland-and-qt)
>     
> *   [Repeated storage of graphical resources](#wayland-and-qt)
>     
> *   [Input latency](#wayland-and-qt)
>     
> 
> Increased video memory consumption
> 
> This can be a constraint for embedded devices. In multi-process, each client needs to have its own graphics buffer, which it sends to the compositor. Consequently, you use more video memory compared to the single-process case: where everything is drawn at once and there is no need to store the different parts in intermediary buffers.
> 
> Increased main memory consumption
> 
> Apart from some extra overhead at the OS level, running multiple clients may also use more main memory as some parts need to be duplicated once per client. For example, if you run QML, each client requires a separate QML engine. Consequently, if you run a single client that uses Qt Quick Controls, it’s loaded once. If you then split this client into multiple clients, you’re loading Qt Quick Controls multiple times, resulting in a higher startup cost to initialize your clients.
> 
> Repeated storage of graphical resources
> 
> In a single-process system, if you’re using the same textures, background, or icons in many places, those images are only stored once. In contrast, if you use these images in a multi-process system, then you have to store them multiple times. In this case, one solution is to share graphical resource between clients. Qt already allows sharing image resources in main memory across processes without involving Wayland. Sharing GPU textures across processes, on the other hand, requires more intricate solutions. With Qt, such solutions can be developed as Wayland extension protocols and with QQuickImageProvider, for instance.
> 
> Input-to-photon latency
> 
> On a single-process system, the application accesses the main frame buffer directly. This means that the latency between input events and reflecting them on screen can be minimized in such a setup. On a multi-process system, the application content has to be triple-buffered to ensure the client isn’t drawing into the buffers while they are simultaneously being read by the server, as that would cause tearing. This means that there is an implicit latency in a multi-process system.

Why Use Wayland Instead of X11 or Custom Solutions[¶](#why-use-wayland-instead-of-x11-or-custom-solutions "Link to this heading")
---------------------------------------------------------------------------------------------------------------------------------

As described earlier, X11 is not an optimal match for typical system setups today. It is quite large and complex, and lacks ability in customization. In fact, it is difficult to run a client fluidly with X11, and reach 60 fps without tearing. Wayland, in contrast, is easier to implement, has better performance, and contains all the necessary parts to run efficiently on modern graphics hardware. For embedded, multi-process systems on Linux, Wayland is the standard.

However, if you are working with old hardware or legacy applications, then Wayland may not be a good option. The Wayland protocol is designed with security and isolation in mind, and is strict/conservative about what information and functionality is available to clients. While this leads to a cleaner and more secure interface, some functionality that legacy applications expect may no longer be available on Wayland.

Particularly, there are three common use cases where Wayland may not be the best option:

> 1.  The hardware or platform is old and only supports X11; in which case you have no choice.
>     
> 2.  You have to support legacy applications that depend on features that are absent in the Wayland protocol for security and simplicity.
>     
> 3.  You have to support legacy applications that use a UI toolkit that doesn’t run on Wayland at all. In some cases, you may be able to work around this by running those applications on [XWayland](https://wayland.freedesktop.org/docs/html/ch05.html) instead.
>     

Back when X11 was very popular, developers wrote their own custom solutions to circumvent X11 issues. Older Qt versions had the Qt Windowing System (QWS), which is now discontinued. Today, most of these use cases are covered by Wayland, and custom solutions are becoming less and less common.

What Qt Wayland Offers[¶](#what-qt-wayland-offers "Link to this heading")
-------------------------------------------------------------------------

**For Clients** Qt clients can run on any Wayland compositor, including Weston, the reference compositor developed as part of the Wayland project.

Any Qt program can run as a Wayland client (as part of a multi-process system) or a standalone client (single-process). This is determined on startup, where you can choose between the different backends. During the development process, you can develop the client on the desktop first, then test it on the target hardware later. You don’t need to run your clients on the actual target hardware all the time.

> ![../_images/wayland-single-process-develop.png](../_images/wayland-single-process-develop.png)

Single-Process Client Development

If you develop on a Linux machine, you can also run the compositor within a window on your development machine. This lets you run clients in an environment that closely resembles the target device. Without rebuilding the client, you can also run it with `-platform wayland` to run it inside the compositor. If you use `-platform xcb` (for X11), you can run the client on the desktop. In other words, you can start developing your clients before the compositor is ready for use.

**For Servers** The server, or compositor, connects to the display and shows the contents of each client on the screen. The compositor handles input and sends input events to the corresponding client. In turn, each client connects to the compositor and sends the content of its windows. It’s up to the compositor to decide:

> *   How and where to show the content
>     
> *   Which content to show
>     
> *   What to do with the different client graphics buffers
>     

This means, it’s up to the compositor to decide what a multi-process system is. For instance, the clients could be part of a 3D scene with windows on the walls, on a VR system, mapped to a sphere, and so on.

The Qt Wayland Compositor is an API for building your own compositor. It gives you full freedom to build a custom compositor UI and manage the windows of various clients. You can combine both Qt Quick and QML with the Qt Wayland Compositor to create impressive, imaginative UIs. For more information, see Qt Wayland Compositor.

Qt also provides powerful and user-friendly APIs to implement Wayland extensions and use them from QML or C++.

Related Content[¶](#related-content "Link to this heading")
-----------------------------------------------------------

> *   [QtWS17 - Qt Wayland Compositor: Creating multi-process user interface](https://resources.qt.io/videos/qtws17-qt-wayland-compositor-creating-multi-process-user-interface-johan-helsing-the-qt-company)
>     
> *   [Qt Application Manager](https://doc.qt.io/QtApplicationManager/introduction.html)
>     

Copyright © 2025 The Qt Company Ltd. Documentation contributions included herein are the copyrights of their respective owners. The documentation provided herein is licensed under the terms of the GNU Free Documentation License version 1.3 (https://www.gnu.org/licenses/fdl.html) as published by the Free Software Foundation. Qt and respective logos are trademarks of The Qt Company Ltd. in Finland and/or other countries worldwide. All other trademarks are property of their respective owners.

Multi-Process or Single-Process
When building a simple embedded platform with Qt, a perfectly viable option is to have all parts of the UI running in a single process. However, as the system becomes more complex, you may want to consider a multi-process system instead. This is where Wayland comes in. With Qt, at any point in your development process, you can choose to switch between single-process and multi-process.

Benefits of Multi-Process
The following diagrams illustrate the difference between multi-process and single-process systems.

../_images/wayland-multi-process.png
Multi-Process Client Architecture

../_images/wayland-single-process-eglfs.png
Single Process Client Architecture

The Qt Wayland Compositor module is ideal for creating the display server and compositor in multi-process systems on embedded Linux. The use of multi-process has the following benefits:

Stability

Security

Performance

Interoperability

Stability

Easier to recover when clients hang or crash

If you have a complex UI, then multi-process is useful because if one part of the UI crashes, it doesn’t affect the entire system. Similarly, the display won’t freeze, even when one client freezes.

Note

If your client is mandated by law to render safety-critical information, consider using Qt Safe Renderer Overview .

Protection against possible memory leaks

In a multi-process system, if one client has a memory leak and consumes lots of memory, that memory is recovered when that client exits. In contrast with single-process, the memory leak remains until the entire system restarts.

Security

In a single-process system, all clients can access each other’s memory. For example, there’s no isolation for sensitive data transfer; every line of code must be equally trustworthy. This isolation is there, by design, in multi-process systems.

Performance

If you have a CPU with multiple cores, a multi-process system can help distribute the load evenly across different cores, making more efficient use of your CPU.

Interoperability

You can interface with non-Qt clients in a multi-process system, as long as your clients understand Wayland or X11. For example, if you use gstreamer for video or if you want to use a navigation application built with another UI toolkit, you can run these clients alongside your other Qt-based clients.

Trade-offs of Multi-Process
When going from single-process to multi-process, it is important to be conscious of the following trade-offs:

Increased video memory consumption

Increased main memory consumption

Repeated storage of graphical resources

Input latency

Increased video memory consumption

This can be a constraint for embedded devices. In multi-process, each client needs to have its own graphics buffer, which it sends to the compositor. Consequently, you use more video memory compared to the single-process case: where everything is drawn at once and there is no need to store the different parts in intermediary buffers.

Increased main memory consumption

Apart from some extra overhead at the OS level, running multiple clients may also use more main memory as some parts need to be duplicated once per client. For example, if you run QML, each client requires a separate QML engine. Consequently, if you run a single client that uses Qt Quick Controls, it’s loaded once. If you then split this client into multiple clients, you’re loading Qt Quick Controls multiple times, resulting in a higher startup cost to initialize your clients.

Repeated storage of graphical resources

In a single-process system, if you’re using the same textures, background, or icons in many places, those images are only stored once. In contrast, if you use these images in a multi-process system, then you have to store them multiple times. In this case, one solution is to share graphical resource between clients. Qt already allows sharing image resources in main memory across processes without involving Wayland. Sharing GPU textures across processes, on the other hand, requires more intricate solutions. With Qt, such solutions can be developed as Wayland extension protocols and with QQuickImageProvider, for instance.

Input-to-photon latency

On a single-process system, the application accesses the main frame buffer directly. This means that the latency between input events and reflecting them on screen can be minimized in such a setup. On a multi-process system, the application content has to be triple-buffered to ensure the client isn’t drawing into the buffers while they are simultaneously being read by the server, as that would cause tearing. This means that there is an implicit latency in a multi-process system.

Why Use Wayland Instead of X11 or Custom Solutions
As described earlier, X11 is not an optimal match for typical system setups today. It is quite large and complex, and lacks ability in customization. In fact, it is difficult to run a client fluidly with X11, and reach 60 fps without tearing. Wayland, in contrast, is easier to implement, has better performance, and contains all the necessary parts to run efficiently on modern graphics hardware. For embedded, multi-process systems on Linux, Wayland is the standard.

However, if you are working with old hardware or legacy applications, then Wayland may not be a good option. The Wayland protocol is designed with security and isolation in mind, and is strict/conservative about what information and functionality is available to clients. While this leads to a cleaner and more secure interface, some functionality that legacy applications expect may no longer be available on Wayland.

Particularly, there are three common use cases where Wayland may not be the best option:

The hardware or platform is old and only supports X11; in which case you have no choice.

You have to support legacy applications that depend on features that are absent in the Wayland protocol for security and simplicity.

You have to support legacy applications that use a UI toolkit that doesn’t run on Wayland at all. In some cases, you may be able to work around this by running those applications on XWayland instead.

Back when X11 was very popular, developers wrote their own custom solutions to circumvent X11 issues. Older Qt versions had the Qt Windowing System (QWS), which is now discontinued. Today, most of these use cases are covered by Wayland, and custom solutions are becoming less and less common.

What Qt Wayland Offers
For Clients Qt clients can run on any Wayland compositor, including Weston, the reference compositor developed as part of the Wayland project.

Any Qt program can run as a Wayland client (as part of a multi-process system) or a standalone client (single-process). This is determined on startup, where you can choose between the different backends. During the development process, you can develop the client on the desktop first, then test it on the target hardware later. You don’t need to run your clients on the actual target hardware all the time.

../_images/wayland-single-process-develop.png
Single-Process Client Development

If you develop on a Linux machine, you can also run the compositor within a window on your development machine. This lets you run clients in an environment that closely resembles the target device. Without rebuilding the client, you can also run it with -platform wayland to run it inside the compositor. If you use -platform xcb (for X11), you can run the client on the desktop. In other words, you can start developing your clients before the compositor is ready for use.

For Servers The server, or compositor, connects to the display and shows the contents of each client on the screen. The compositor handles input and sends input events to the corresponding client. In turn, each client connects to the compositor and sends the content of its windows. It’s up to the compositor to decide:

How and where to show the content

Which content to show

What to do with the different client graphics buffers

This means, it’s up to the compositor to decide what a multi-process system is. For instance, the clients could be part of a 3D scene with windows on the walls, on a VR system, mapped to a sphere, and so on.

The Qt Wayland Compositor is an API for building your own compositor. It gives you full freedom to build a custom compositor UI and manage the windows of various clients. You can combine both Qt Quick and QML with the Qt Wayland Compositor to create impressive, imaginative UIs. For more information, see Qt Wayland Compositor.

Qt also provides powerful and user-friendly APIs to implement Wayland extensions and use them from QML or C++.


QtWS17 - Qt Wayland Compositor: Creating multi-process user interface, Johan Helsing, The Qt Company
This talk gives an in-depth presentation of the Qt Wayland Compositor API, showing you how to create a custom Wayland compositor. We will start from scratch and create our own user interface with animations, bells and whistles. As the complexity and requirements of user interfaces grow, it becomes natural to split the UI into several processes. Doing so can improve stability, security, and enable 3rd party application development. Qt has made the decision to use Wayland for this purpose. With Qt Wayland Compositor you can write your own Wayland compositor, making it possible to create custom user interfaces for multi-process embedded applications.


Introduction
The Qt Application Manager is a headless daemon that helps you to create embedded Linux systems with a highly complex UI setup, which you can optionally split into a multi-process setup to increase flexibility and stability.



Qt Application Manager Architecture

The main building blocks of the application manager are:

Wayland Window Compositor
Application Launcher
User Input Management
Notifications
Application Installation
Combining these building blocks has certain advantages, as described below.

Wayland Window Compositor
To support multiple UI processes on an embedded Linux system, you need a central window compositor: a Wayland compositor is the state-of-the-art solution for this. Consequently, the application manager incorporates a compositor that is fully-compliant with the Wayland protocol, based on the QtWayland module.

The window compositing part is project-specific, that you can write using QtQuick elements, giving you all of QML's capabilities to implement rich and fluid animations in your compositor.

In addition to the Qt Wayland Compositor, the application manager also provides an interface to which you can attach arbitrary meta-data to any Wayland window. This interface is particularly useful on custom embedded UI systems, where the top-level UI elements may not follow the classic desktop-centric Application Window, Modal Dialog, and Popup Menu window classification approach.

Application Launcher
The launcher part is the central component for application life-cycle management: it starts and stops applications (internal or third-party) either on an explicit request or as an automatic reaction to external triggers. For example, in low-memory situations, you want to gracefully terminate applications that the user hasn't interacted with in a while.

Since UI applications typically have to be stopped and restarted multiple times during the up-time of an embedded system, and given the fact that most customers have tight constraints on the startup times of UI applications, the application manager implements some tricks to improve the startup performance: any application type that relies on a runtime component, such as QML or HTML, can be quick-launched.

The actual implementation depends on the type of runtime. The application manager ships with a QML quick-launcher that you can configure to keep at least one process with a fully-initialized QML engine available in the background (dependent on actual CPU load). In this case, starting an application only requires you to load the application's QML files into the existing engine.

Support for other runtimes can be added via an external, independent runtime launcher binary.

In addition to the runtime abstraction, the application manager is also able to run any application inside a container instead of just an external Unix process. Support for these container-based solutions must be provided by the customer and could range from full container solutions like KVM or XEN to LXC or even down to mandatory access control frameworks like AppArmor or SELinux. The application manager includes an integration with bubblewrap containers.

User Input Management
Since most embedded UI applications rely on a virtual keyboard, the Qt Virtual Keyboard module can be integrated into the System UI and the Compositor. Through the Wayland protocol, this virtual keyboard component can then be transparently used from any Qt/QML application for full internationalized text input, without any special provisions on the application side. In contrast, non-Qt applications need to provide support for the required, open Wayland text input protocol.

Notifications
The application manager acts as a freedesktop.org standards-compliant notification server on the D-Bus. For QtQuick applications, a QML component is provided which encapsulates the client side of the freedesktop.org notification protocol.

Both, the client and server sides, also come with Qt-like APIs to extend any notification request with additional meta-data using standard-compliant mechanism.

Application Installation
In addition to built-in applications, that are part of the System UI or the base installation, the application manager also supports third-party applications, that are dynamically installed, updated, and uninstalled.

Developing a built-in application compared to a third-party application is not much different, except for the additional packaging step for third-party applications. The application manager comes with a simple package format as well as a tool to create and digitally sign these packages.

The application manager's installer component can install and update these packages in a background thread using either plain HTTP(S) transfers or by receiving data from a customer-specific local socket connection. To support devices with limited disk space, all package installations and updates take place as they are downloaded to the device. This is more efficient than downloading the entire package first, and then proceeding with the installation or update.

Using a custom package format instead of a standard Unix format is deliberate for two reasons: they either use libraries that support legacy formats or they include hooks for arbitrary shell scripts; these hooks could be used to circumvent security measures.

Instead, the application manager uses a TAR archive with further restrictions on the type of content and some named YAML metadata files, that are application manager-specific. These packages are parsed via the BSD standard libarchive library, which is also the basis for the actual tar command.

Additionally, these packages can be cryptographically signed by both the developer and the production app-store server. The devices' "developer mode" allows using only a developer signature or disabling this signature verification completely.

Advantages
Aggregating all these building blocks into a single daemon enables them to work together much more efficiently:

The compositor can perform security checks when an application requests to show a window, as it has access to the process IDs of all applications started by the application manager. By default, windows from unknown processes are not shown on the screen.
The application manager enforces policies on the usage of its internal IPC mechanism as well as provides a D-Bus interface to the system's middleware to allow other process and libraries to authenticate resource usage of applications started by the application manager. This is possible because of the application's capabilities, that are stored in digitally signed manifests, and the application manager's knowledge of the application-to-PID mappings.
The application manager's ability to run the same System UI and QML applications in both single- and multi-process mode also has quite some advantages - both during the development phase as well as for product scaling. Typically, this does not require any changes to the System UI or the QML applications themselves.

As a developer you can choose which desktop OS you want to develop on. For single-process testing you can choose from Windows, macOS, or Linux; without requiring Wayland. For multi-process, you can choose between Linux or macOS. On Linux, the application manager uses nested Wayland. On macOS, Wayland support is experimental.
Both modes can be mixed, for example, third–party applications could be run in separate processes (or even in a secure container), while built-in QML applications could be loaded in-process. This configuration results in quicker startup times.
The UI code that runs on the target system is the same code that runs on the developers' machines.
The application manager gives you the possibility to scale down your product to lower-end hardware by saving on system and graphics memory, as well as startup times. You can do this by moving either all or just a few critical applications from multi- to single-process mode.
Be aware that any non-QML application, such as a native C++ compiled executable, will break this setup for developers on machines without Wayland support.

Still, there are projects that require applications using multiple UI technologies like QML, HTML, or native OpenGL rendering. In this scenario, the application manager's ability to support various runtimes makes it possible to composite all these applications seamlessly into a consistent user experience.

PySide6.QtConcurrent
Detailed Description
The Qt Concurrent module contains functionality to support concurrent execution of program code.

The Qt Concurrent module provides high-level APIs that make it possible to write multi-threaded programs without using low-level threading primitives such as mutexes, read-write locks, wait conditions, or semaphores. Programs written with Qt Concurrent automatically adjust the number of threads used according to the number of processor cores available. This means that applications written today will continue to scale when deployed on multi-core systems in the future.

Qt Concurrent includes functional programming style APIs for parallel list processing, including a MapReduce and FilterReduce implementation for shared-memory (non-distributed) systems, and classes for managing asynchronous computations in GUI applications:

QFuture represents the result of an asynchronous computation.

QFutureIterator allows iterating through results available via QFuture .

QFutureWatcher allows monitoring a QFuture using signals-and-slots.

QFutureSynchronizer is a convenience class that automatically synchronizes several QFutures.

QPromise provides a way to report progress and results of the asynchronous computation to QFuture . Allows suspending or canceling the task when requested by QFuture .

Using the Module
To include the definitions of modules classes, use the following directive:



How do I properly perform multiprocessing from PyQt?

Bendegúz Szatmári already answer the main question.

I just want to let you know that use Process is not best idea in most of usage. Different process does not share memory with your program. You can not control them so easily as different thread.

Here is simple example how you can Start end Stop different thread.

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
import sys
import time


class TTT(QThread):
    def __init__(self):
        super(TTT, self).__init__()
        self.quit_flag = False

    def run(self):
        while True:
            if not self.quit_flag:
                self.doSomething()
                time.sleep(1)
            else:
                break

        self.quit()
        self.wait()

    def doSomething(self):
        print('123')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.btn = QtWidgets.QPushButton('run process')
        self.btn.clicked.connect(self.create_process)
        self.setCentralWidget(self.btn)

    def create_process(self):
        if self.btn.text() == "run process":
            print("Started")
            self.btn.setText("stop process")
            self.t = TTT()
            self.t.start()
        else:
            self.t.quit_flag = True
            print("Stop sent")
            self.t.wait()
            print("Stopped")
            self.btn.setText("run process")


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())



-----------------------

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



About Valkka
Why this library?
So, yet another media player? I need to stream video from my IP camera into my python/Qt program and I want something that can be developed fast and is easy to integrate into my code. What’s here for me?

If you just need to stream video from your IP cameras, decode it and show it on the screen, we recommend a standard media player, say, VLC and its python bindings.

However, if you need to stream video and simultaneously (1) present it on the screen, (2) analyze it with machine vision, (3) write it to disk, and even (4) recast it to other clients, stock media players won’t do.

Such requirements are typical in large-scale video surveillance, management and analysis solutions. Demand for them is growing rapidly due to continuous decline in IP camera prices and growing computing power.

As a solution, you might try connect to the same camera 4 times and decode the stream 4 times - but then you’ll burn all that CPU for nothing (you should decode only once). And try to scale that only to, say, 20+ cameras. In order avoid too many connections to your IP cameras (this is typically limited by the camera), you might desperately try your luck even with the multicast loopback. We’ve been there and it’s not a good idea. And how about pluggin in your favorite machine vision/learning module, written with OpenCV or TensorFlow?

Valkka API
Valkka will solve the problem for you; It is a programming library and an API to do just that - large scale video surveillance, management and analysis programs, from the comfort of python3.

With Valkka, you can create complex pipings (“filtergraphs”) of media streams from the camera, to screen, machine vision subroutines, to disk, to the net, etc. The code runs at the cpp level with threads, thread-safe queues, mutexes, semaphores, etc. All those gory details are hidden from the API user that programs filtergraphs at the python level only. Valkka can also share frames between python processes (and from there, with OpenCV, TensorFlow, etc.)

If you got interested, we recommend that you do the tutorial, and use it together with the PyQt testsuite, and the example project as starting points for your own project.

This manual has a special emphasis for Qt and OpenCV. You can create video streaming applications using PyQt: streaming video to widgets, and connect the signals from your machine vision subprograms to the Qt signal/slot system - and beyond.

For more technical information, check out the library architecture page

Finally, here is a small sample from the tutorial. You’ll get the idea.

main branch, streaming
(LiveThread:livethread) --> ----------------------------------+
                                                              |
                                                              |
{ForkFrameFilter: fork_filter} <----(AVThread:avthread) << ---+  main branch, decoding
               |
      branch 1 +-->> (OpenGLThread:glthread) --> To X-Window System
               |
      branch 2 +---> {IntervalFrameFilter: interval_filter} --> {SwScaleFrameFilter: sws_filter} --> {RGBSharedMemFrameFilter: shmem_filter}
                                                                                                                  |
                                                                                              To OpenCV  <--------+
The Project
In Valkka, the “streaming pipeline” from IP cameras to decoders and to the GPU has been completely re-thinked and written from scratch:

No dependencies on external libraries or x window extensions (we use only glx)

Everything is pre-reserved in the system memory and in the GPU. During streaming, frames are pulled from pre-reserved stacks

OpenGL pixel buffer objects are used for texture video streaming (in the future, we will implement fish-eye projections)

Customized queueing and presentation algorithms

etc., etc.

Valkka is in alpha stage. Even so, you can do lot of stuff with it - at least all the things we have promised here in the intro.

Repositories are organized as follows:

valkka-core : the cpp codebase and its python bindings are available at the valkka-core github repository. The cpp core library is licensed under LGPL license see here.

valkka-examples : the python tutorial and PyQt example/testsuite are available at the valkka-examples github repository. MIT licensed.

For more, see here.

All functional features are demonstrated in the tutorial which is updated as new features appear. Same goes for the PyQt testsuite.

Near-term goals for new features are:

Interserver communications between Valkka-based server and clients

ValkkaFS filesystem, designed for recording large amounts of video (not yet fully operational / debugged)

Synchronized recording of video

Fisheye projections

Support for sound

Valkka is based on the following opensource libraries and technologies:
https://elsampsa.github.io/valkka-examples/_build/html/intro.html


---------------_

Doing python multiprocessing The Right Way
Sampsa Riikonen
Sampsa Riikonen

Follow
7 min read
·
Jul 27, 2021
265


1





Learn how to combine multiprocessing- and threading, and how to organize your multiprocessing classes in the right way.

Press enter or click to view image in full size

A Posix Fork has taken place
Not a day goes by in Medium articles without someone complaining that Python is not the future of machine learning.

For example, things like “I can write a GPU kernel in Julia but not in Python”. However, most of us data scientists/engineers are just dummy engineers who use ready-made libraries. 95% of us are not interested in “writing a GPU kernel”.

Another complaint is about multiprocessing: on many occasions, you need to use multiprocessing instead of multithreading. People find this awkward(especially because of interprocess communication). Also, multiprocessing is very prone to errors if you’re not carefull.

Although cumbersome at first sight, multiprocessing does have several advantages.

Keeping processes separated in their own (virtual memory) “cages” can actually help in debugging and avoids confusion. Once you sort out the intercommunication problem in a systematic way and avoid some common pitfalls, programming with python multiprocesses becomes a joy.

One frequent error is to mix multithreading and multiprocessing together, creating a crashy/leaky program and then conclude that python sucks. More on this later.

Some beginners prefer, instead of writing a proper multiprocessing class, to do things like this:

p = Process(target=foo)
This obfuscates completely what you are doing with processes and threads (see below).

I have even seen people using multiprocessing.Pool to spawn single-use-and-dispose multiprocesses at high frequency and then complaining that "python multiprocessing is inefficient".

After this article you should be able to avoid some common pitfalls and write well-structured, efficient and rich python multiprocessing programs.

This is going to be different what you learned in that python multiprorcessing tutorial. No Managers, Pools or Queues, but more of an under-the-hood approach.

Let’s start with the basics of posix fork.

Forking vs. Threading
Forking/multiprocessing means that you “spawn” a new process into your system. It runs in its own (virtual) memory space. Imagine that after a fork, a copy of your code is now running “on the other side of the fork”. Think of “Stranger Things”.

On the contrary, threading means you are creating a new running instance/thread, toiling around in the same memory space with your current python process. They can access the same variables and objects.

Confusing bugs arise when you mix forking and threading together, as creating threads first and then forking, leaves “dangling”/”confused” threads in the spawned multiprocesses. Talk about mysterious freezes, segfaults and all that sort of nice things. But combining forking and threading can be done, if it’s done in the right order: fork first and then threading.

This problem is further aggravated by the fact that many libraries which you use in your python programs may start threads sneakily in the background (many times, directly in their C source code), while you are completely unaware of it.

Said all that, this is the correct/safe order of doing things:

0. Import libraries that do not use multithreading
1. Create interprocess communication primitives and shared resources that are shared between multiprocesses (however, not considered in this tutorial)
2. Create interprocess communication primitives and shared resources that are shared with the main process and your current multiprocess
3. Fork (=create multiprocesses)
4. Import libraries that use multithreading
5. If you use asyncio in your multiprocess, create a new event loop
Let’s blend these steps in with an actual code:


Remember that concept of code running “on the other side of the fork”? That “other side” with demogorgons (and the like) which is isolated from our universe is created when you say p.start().

The stuff that runs in that parallel universe is defined in the method run().

When creating complex multiprocessing programs, you will have several multiprocesses (parallel universes) each one with a large codebase.

So, we’ll be needing a “mental guideline” to keep our mind in check. Let’s introduce a concept for that purpose:

Our multiprocess class shall have a frontend and a backend (not to be confused with web development!)

Frontend is the scope of your current running python interpreter. The normal world.

Backend is the part of the code that runs “on the other side of the fork”. It’s a different process in its own memory space and universe. Frontend needs to communicate with the backend in some way (think again of Stranger Things).

Let’s once more emphasize that everything that’s inside/originates from method run(), runs in the backend.

From now on, we’ll stop talking about demogorgons, parallel realities and stick strictly to frontend and backend. Hopefully, you have made the idea by now.

The only things happening at the frontend in that example code are:

p = MyProcess() # code that is executed in MyProcess.__init__

p.start() # performs the fork

In order to avoid confusion, we need to differentiate between frontend and backend methods. We need a naming convention. Let’s use this one:

All backend methods shall have a double-underscore in their name

Like this:


i.e. listenFront__() is a backend method.

Before we move on, one extra observation: multiprocesses are not supposed to be single-use-and-dispose. You don’t want to create and start them at high frequency since creating them has considerable overhead. You should try to spawn your multiprocesses only once (or at very low frequency).

Let’s Ping Pong
Next, let’s demonstrate the frontend/backend scheme in more detail.

We do a classical multiprocessing example: sending a ping to the multiprocess, which then responds with a pong.

The frontend methods are ping() and stop(). You call these methods in your main python program (aka frontend). Under-the-hood, these methods do seamless intercommunication between front- and backend.

Backend methods listenFront__() and ping__() run at the backend and they originate from the run() method.

Here’s the code:


Note that in the python main process, we use only the frontend methods (start, ping and stop).

So, we have successfully eliminated the mental load of needing to think about the fork at all. At the same time, the code has a clear distinction to and intercommunication with the forked process. We just need to think in terms of the front- and backend and their corresponding methods.

One more pitfall
Let’s imagine that, as your codebase grows, your code looks something like this:


SomeLibrary is just some library that you need in your code but is not used/related to your multiprocesses in any way.

However, if that SomeLibrary uses multithreading under-the-hood (without you knowing about it), you have created yourself a big problem.

Still remember what we said earlier?

No threads before fork!

As even just importing a library might silenty starts threads, to be absolutely on the safe side, do this instead:


i.e. instantiate and start your multiprocesses before anything else.

If the logic in your program requires using multiprocesses “on-demand”, consider this:


i.e., instead of creating and starting multiprocesses in the middle of your program, you create and start them at the very beginning and then cache them for future use.

Some Testing and debugging tips
For test purposes, you can run your python multiprocessing classes without forking at all, by simply not using start()in your test code. In this case you can call the backend methods directly in your tests/frontend, provided that you have structured your code correctly.

For python refleaks and resulting memory blowup issues you can use the following technique. Import the setproctitle library with

from setproctitle import setproctitle
In your multiprocesses run() method, include this:

setproctitle("Your-Process-Name")
Now your process is tagged with a name, so that you can follow the memory consumption of that single process very easily with standard linux tools, say, with smem and htop (in htop, remember to go to setup => display options and enable "Hide userland process threads" in order to make the output more readable).

Finally
In this tutorial I have given you some guidelines to succeed with your python multiprocessing program and not to fall into some typical pitfalls.

You might still have lot of questions:

How to listen at several multiprocesses simultaneously at my main program? (hint: use the select module)
How do I send megabytes of streaming data to a running multiprocess? Say, images and/or video (can be done perfectly, but not trivial)
Can I run asyncio in the back- or frontend or both? (sure)
These are, however, out of the scope of this tutorial.

Let’s just mention that in the case (2) that:

You would not certainly use pipes (they are not for large streaming data)
Use posix shared memory, mapped into numpy arrays instead. Those arrays form a ring-buffer that is synchronized using posix semaphores across process boundaries
You need to listen simultaneously to the intercommunication pipe and the ringbuffer. Posix EventFd is a nice tool for this.
I’ve done this kind of stuff in a python multimedia framework I’ve written. If you’re interested, please see here.

That’s the end of the tutorial. I hope it gave you some food for thought.

APPENDUM 1

I wrote an addinitional example, available here:

Posix shared memory and intercommunicating with several multiprocesses (questions 1 & 2 above).
Special interest for PyQt / PySide2 users: mapping slots and signals to a multiprocess.
Handling multiprocesses from a multithread
APPENDUM 2

All the ideas presented in this article, are now neatly wrapped into a python module: please see here

Valkka Multiprocess
Advanced Python Multiprocessing and Inter-Process Communication

Contents:

Intro
Installing
Tutorial
Part I: Managing a multiprocess
1. Send message to a forked process
2. Send message and synchronize
3. Send and receive message
4. Using shared memory and forked resources
5. Syncing server/client resources
Part II: Organizing workers
6. Planning it
7. Implementation
Part III: Miscellaneous
Development cleanup
Process debug
Streaming data
Interfacing with C++
Custom MessageProcess
Qt related
Asyncio
API documentation
MessageObject
MessageObject
MessageProcess
MessageProcess
MessageProcess.c__ping()
MessageProcess.formatLogger()
MessageProcess.getPipe()
MessageProcess.go()
MessageProcess.ignoreSIGINT()
MessageProcess.postRun__()
MessageProcess.preRun__()
MessageProcess.readPipes__()
MessageProcess.requestStop()
MessageProcess.run()
MessageProcess.sendMessageToBack()
MessageProcess.sendPing()
MessageProcess.send_out__()
MessageProcess.stop()
MessageProcess.waitStop()
AsyncBackMessageProcess
AsyncBackMessageProcess
AsyncBackMessageProcess.asyncPost__()
AsyncBackMessageProcess.asyncPre__()
AsyncBackMessageProcess.c__ping()
AsyncBackMessageProcess.getPipe()
AsyncBackMessageProcess.getReadFd()
AsyncBackMessageProcess.getWriteFd()
AsyncBackMessageProcess.send_out__()
MainContext
MainContext
MainContext.__call__()
MainContext.close()
MainContext.formatLogger()
MainContext.runAsThread()
MainContext.startProcesses()
MainContext.startThreads()
MainContext.stopThread()
EventGroup
EventGroup
EventGroup.asIndex()
EventGroup.fromIndex()
EventGroup.release()
EventGroup.release_ind()
EventGroup.reserve()
EventGroup.set()
SyncIndex
SyncIndex
Other
safe_select()
https://medium.com/@sampsa.riikonen/doing-python-multiprocessing-the-right-way-a54c1880e300
-------------------


lassoan
Andras Lasso
May 2022
Most objects are only usable on a single thread. Therefore, you usually cannot simply run the same code in a processing thread that works on the main thread.

You can only run processing in the worker threads and you must not access GUI widgets and application objects. See GitHub - pieper/SlicerParallelProcessing: Slicer modules for running subprocesses to operate on data in parallel extension for examples of how you can implement parallel processing in Python in Slicer.

https://github.com/pieper/SlicerParallelProcessing
ParallelProcessing
ParallelProcessing extension for 3D Slicer. Currently contains Processes module.

Processes
Slicer modules for running subprocesses to operate on data in parallel.

(Still a work in progress, but starting to be useable)

In particular, this is designed to work with python scripts that do headless processing (no graphics or GUI).

Installation
Simply run slicer with these arguments:

./path/to/Slicer --additional-module-paths path/to/SlicerProcesses/Processes

where 'path/to' is replaced with the appropriate paths. You could alternatively register the path in the Module paths of the Application Settings dialog.

Usage
This is designed to be a developer tool, so look at the Model and Volume tests at the bottom of the module source code.

The basic idea is:

you write a processing python script that does the following
reads binary data from stdin
unpickles that data to make python object (e.g. dictionary of parameters such as numpy arrays)
uses that data as needed
builds an output dictionary of results
pickles the output and writes the binary to stdout
within your Slicer module you do the following
create a subclass of the Process class
add whatever parameters are needed in the constructor
override the prepareProcessInput method to create the format your processing script expects Ue.g. picked dict)
override the useProcessOutput method to consume what your script creates
use the ProcessingLogic class to add instances of your class and trigger them to run
use the completedCallback function to trigger next steps when processes have all finished
optionally block using waitForFinished (don't use this unless you really need to since it is a blocking busy loop)
for finer grained status updates observe the moduule node as is done in the gui widget of the module
Demo
Here's what the self-test looks like. What happens is that a dummy sphere is added to the scene 50 processes are queued, each of which applies a random offet to each of the vertices. The machine running the test has 12 cores, so you see the processes being executed aproximately in groups of 12. The second part shows running 5 parallel image filtering operations with different filter kernel radius values and then loading the results. Code for these demos is in this repository.

IMAGE ALT TEXT

Future Directions
GUI
right now the gui just shows the status of processes, but it could be made more useful to show how long a process has been running or other stats like memory consumption
it could be useful to be able to cancel a process from the gui
there's currently no way to clear the output list of completed processes
Logic
Other than limiting the number of running processes there's no way to load balance
On the whole though it's good that the logic class is very clean and short so we shouldn't overcomplicate it
Architecture and style
Could be good to break up the code into multiple files if it gets much longer
A helper package to pickle vtk and mrml classes would be nice, but it should be independent of this module
Process input/output shouldn't be restricted to only pickling, any ascii or binary data would work
Some more worked out examples of different use cases could help confirm that the design is workable
Additional functionality directions
A process could be kept alive and exchange multiple messages (may not be worth the complexity)
Process invocations could be wrapped in ssh for remote execution on cluster or cloud compute resources. The remote account would only need to have a compatible installation of Slicer (PythonSlicer) in the path.
Cloud computing resources (virtual machines) could even be created on the fly to perform bigger jobs
The processes don't need to only operate on data that exists in Slicer, but instead the processes could download from URLs and similarly could upload results elsewhere; this could be useful in the case where Slicer's UI is used to provide interactive inputs or spot check and QC results.
Future






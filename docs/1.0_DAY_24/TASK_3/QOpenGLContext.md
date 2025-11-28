PySide6.QtGui.QOpenGLContext
class QOpenGLContext
The QOpenGLContext class represents a native OpenGL context, enabling OpenGL rendering on a QSurface . More…


Synopsis
Methods
def __init__()

def create()

def defaultFramebufferObject()

def doneCurrent()

def extensions()

def extraFunctions()

def format()

def functions()

def getProcAddress()

def hasExtension()

def isOpenGLES()

def isValid()

def makeCurrent()

def resolveInterface()

def screen()

def setFormat()

def setScreen()

def setShareContext()

def shareContext()

def shareGroup()

def surface()

def swapBuffers()

Signals
def aboutToBeDestroyed()

Static functions
def areSharing()

def currentContext()

def globalShareContext()

def openGLModuleType()

def supportsThreadedOpenGL()

Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

Detailed Description
QOpenGLContext represents the OpenGL state of an underlying OpenGL context. To set up a context, set its screen and format such that they match those of the surface or surfaces with which the context is meant to be used, if necessary make it share resources with other contexts with setShareContext() , and finally call create() . Use the return value or isValid() to check if the context was successfully initialized.

A context can be made current against a given surface by calling makeCurrent() . When OpenGL rendering is done, call swapBuffers() to swap the front and back buffers of the surface, so that the newly rendered content becomes visible. To be able to support certain platforms, QOpenGLContext requires that you call makeCurrent() again before starting rendering a new frame, after calling swapBuffers() .

If the context is temporarily not needed, such as when the application is not rendering, it can be useful to delete it in order to free resources. You can connect to the aboutToBeDestroyed() signal to clean up any resources that have been allocated with different ownership from the QOpenGLContext itself.

Once a QOpenGLContext has been made current, you can render to it in a platform independent way by using Qt’s OpenGL enablers such as QOpenGLFunctions , QOpenGLBuffer, QOpenGLShaderProgram, and QOpenGLFramebufferObject. It is also possible to use the platform’s OpenGL API directly, without using the Qt enablers, although potentially at the cost of portability. The latter is necessary when wanting to use OpenGL 1.x or OpenGL ES 1.x.

For more information about the OpenGL API, refer to the official OpenGL documentation .

For an example of how to use QOpenGLContext see the OpenGL Window example.

Thread Affinity
QOpenGLContext can be moved to a different thread with moveToThread(). Do not call makeCurrent() from a different thread than the one to which the QOpenGLContext object belongs. A context can only be current in one thread and against one surface at a time, and a thread only has one context current at a time.

Context Resource Sharing
Resources such as textures and vertex buffer objects can be shared between contexts. Use setShareContext() before calling create() to specify that the contexts should share these resources. QOpenGLContext internally keeps track of a QOpenGLContextGroup object which can be accessed with shareGroup() , and which can be used to find all the contexts in a given share group. A share group consists of all contexts that have been successfully initialized and are sharing with an existing context in the share group. A non-sharing context has a share group consisting of a single context.

Default Framebuffer
On certain platforms, a framebuffer other than 0 might be the default frame buffer depending on the current surface. Instead of calling glBindFramebuffer(0), it is recommended that you use glBindFramebuffer(ctx-> defaultFramebufferObject() ), to ensure that your application is portable between different platforms. However, if you use glBindFramebuffer() , this is done automatically for you.

Warning

WebAssembly

We recommend that only one QOpenGLContext is made current with a QSurface , for the entire lifetime of the QSurface . Should more than once context be used, it is important to understand that multiple QOpenGLContext instances may be backed by the same native context underneath with the WebAssembly platform. Therefore, calling makeCurrent() with the same QSurface on two QOpenGLContext objects may not switch to a different native context in the second call. As a result, any OpenGL state changes done after the second makeCurrent() may alter the state of the first QOpenGLContext as well, as they are all backed by the same native context.

Note

This means that when targeting WebAssembly with existing OpenGL-based Qt code, some porting may be required to cater to these limitations.

See also

QOpenGLFunctions QOpenGLBufferQOpenGLShaderProgramQOpenGLFramebufferObject

class OpenGLModuleType
This enum defines the type of the underlying OpenGL implementation.

Constant

Description

QOpenGLContext.OpenGLModuleType.LibGL

OpenGL

QOpenGLContext.OpenGLModuleType.LibGLES

OpenGL ES 2.0 or higher

__init__([parent=None])
Parameters:
parent – QObject

Creates a new OpenGL context instance with parent object parent.

Before it can be used you need to set the proper format and call create() .

See also

create() makeCurrent()

aboutToBeDestroyed()
This signal is emitted before the underlying native OpenGL context is destroyed, such that users may clean up OpenGL resources that might otherwise be left dangling in the case of shared OpenGL contexts.

If you wish to make the context current in order to do clean-up, make sure to only connect to the signal using a direct connection.

Note

In Qt for Python, this signal will not be received when emitted from the destructor of QOpenGLWidget or QOpenGLWindow due to the Python instance already being destroyed. We recommend doing cleanups in QWidget::hideEvent() instead.

static areSharing(first, second)
Parameters:
first – QOpenGLContext

second – QOpenGLContext

Return type:
bool

Returns true if the first and second contexts are sharing OpenGL resources.

create()
Return type:
bool

Attempts to create the OpenGL context with the current configuration.

The current configuration includes the format, the share context, and the screen.

If the OpenGL implementation on your system does not support the requested version of OpenGL context, then QOpenGLContext will try to create the closest matching version. The actual created context properties can be queried using the QSurfaceFormat returned by the format() function. For example, if you request a context that supports OpenGL 4.3 Core profile but the driver and/or hardware only supports version 3.2 Core profile contexts then you will get a 3.2 Core profile context.

Returns true if the native context was successfully created and is ready to be used with makeCurrent() , swapBuffers() , etc.

Note

If the context already exists, this function destroys the existing context first, and then creates a new one.

See also

makeCurrent() format()

static currentContext()
Return type:
QOpenGLContext

Returns the last context which called makeCurrent in the current thread, or None, if no context is current.

defaultFramebufferObject()
Return type:
int

Call this to get the default framebuffer object for the current surface.

On some platforms (for instance, iOS) the default framebuffer object depends on the surface being rendered to, and might be different from 0. Thus, instead of calling glBindFramebuffer(0), you should call glBindFramebuffer(ctx->defaultFramebufferObject()) if you want your application to work across different Qt platforms.

If you use the glBindFramebuffer() in QOpenGLFunctions you do not have to worry about this, as it automatically binds the current context’s defaultFramebufferObject() when 0 is passed.

Note

Widgets that render via framebuffer objects, like QOpenGLWidget and QQuickWidget, will override the value returned from this function when painting is active, because at that time the correct “default” framebuffer is the widget’s associated backing framebuffer, not the platform-specific one belonging to the top-level window’s surface. This ensures the expected behavior for this function and other classes relying on it (for example, QOpenGLFramebufferObject::bindDefault() or QOpenGLFramebufferObject::release()).

See also

QOpenGLFramebufferObject

doneCurrent()
Convenience function for calling makeCurrent with a 0 surface.

This results in no context being current in the current thread.

See also

makeCurrent() currentContext()

extensions()
Return type:
.QSetQByteArray

Returns the set of OpenGL extensions supported by this context.

The context or a sharing context must be current.

See also

hasExtension()

extraFunctions()
Return type:
QOpenGLExtraFunctions

Get the QOpenGLExtraFunctions instance for this context.

QOpenGLContext offers this as a convenient way to access QOpenGLExtraFunctions without having to manage it manually.

The context or a sharing context must be current.

The returned QOpenGLExtraFunctions instance is ready to be used and it does not need initializeOpenGLFunctions() to be called.

Note

QOpenGLExtraFunctions contains functionality that is not guaranteed to be available at runtime. Runtime availability depends on the platform, graphics driver, and the OpenGL version requested by the application.

See also

QOpenGLFunctions QOpenGLExtraFunctions

format()
Return type:
QSurfaceFormat

Returns the format of the underlying platform context, if create() has been called.

Otherwise, returns the requested format.

The requested and the actual format may differ. Requesting a given OpenGL version does not mean the resulting context will target exactly the requested version. It is only guaranteed that the version/profile/options combination for the created context is compatible with the request, as long as the driver is able to provide such a context.

For example, requesting an OpenGL version 3.x core profile context may result in an OpenGL 4.x core profile context. Similarly, a request for OpenGL 2.1 may result in an OpenGL 3.0 context with deprecated functions enabled. Finally, depending on the driver, unsupported versions may result in either a context creation failure or in a context for the highest supported version.

Similar differences are possible in the buffer sizes, for example, the resulting context may have a larger depth buffer than requested. This is perfectly normal.

See also

setFormat()

functions()
Return type:
QOpenGLFunctions

Get the QOpenGLFunctions instance for this context.

QOpenGLContext offers this as a convenient way to access QOpenGLFunctions without having to manage it manually.

The context or a sharing context must be current.

The returned QOpenGLFunctions instance is ready to be used and it does not need initializeOpenGLFunctions() to be called.

getProcAddress(procName)
Parameters:
procName – QByteArray

Return type:
QFunctionPointer

Resolves the function pointer to an OpenGL extension function, identified by procName

Returns None if no such function can be found.

getProcAddress(procName)
Parameters:
procName – str

Return type:
QFunctionPointer

static globalShareContext()
Return type:
QOpenGLContext

Returns the application-wide shared OpenGL context, if present. Otherwise, returns None.

This is useful if you need to upload OpenGL objects (buffers, textures, etc.) before creating or showing a QOpenGLWidget or QQuickWidget.

Warning

Do not attempt to make the context returned by this function current on any surface. Instead, you can create a new context which shares with the global one, and then make the new context current.

See also

setShareContext() makeCurrent()

hasExtension(extension)
Parameters:
extension – QByteArray

Return type:
bool

Returns true if this OpenGL context supports the specified OpenGL extension, false otherwise.

The context or a sharing context must be current.

See also

extensions()

isOpenGLES()
Return type:
bool

Returns true if the context is an OpenGL ES context.

If the context has not yet been created, the result is based on the requested format set via setFormat() .

See also

create() format() setFormat()

isValid()
Return type:
bool

Returns if this context is valid, i.e. has been successfully created.

On some platforms the return value of false for a context that was successfully created previously indicates that the OpenGL context was lost.

The typical way to handle context loss scenarios in applications is to check via this function whenever makeCurrent() fails and returns false. If this function then returns false, recreate the underlying native OpenGL context by calling create() , call makeCurrent() again and then reinitialize all OpenGL resources.

On some platforms context loss situations is not something that can avoided. On others however, they may need to be opted-in to. This can be done by enabling ResetNotification in the QSurfaceFormat . This will lead to setting RESET_NOTIFICATION_STRATEGY_EXT to LOSE_CONTEXT_ON_RESET_EXT in the underlying native OpenGL context. QOpenGLContext will then monitor the status via glGetGraphicsResetStatusEXT() in every makeCurrent() .

See also

create()

makeCurrent(surface)
Parameters:
surface – QSurface

Return type:
bool

Makes the context current in the current thread, against the given surface. Returns true if successful; otherwise returns false. The latter may happen if the surface is not exposed, or the graphics hardware is not available due to e.g. the application being suspended.

If surface is None this is equivalent to calling doneCurrent() .

Avoid calling this function from a different thread than the one the QOpenGLContext instance lives in. If you wish to use QOpenGLContext from a different thread you should first make sure it’s not current in the current thread, by calling doneCurrent() if necessary. Then call moveToThread(otherThread) before using it in the other thread.

By default Qt employs a check that enforces the above condition on the thread affinity. It is still possible to disable this check by setting the Qt::AA_DontCheckOpenGLContextThreadAffinity application attribute. Be sure to understand the consequences of using QObjects from outside the thread they live in, as explained in the QObject thread affinity documentation.

See also

functions() doneCurrent() AA_DontCheckOpenGLContextThreadAffinity

static openGLModuleType()
Return type:
OpenGLModuleType

Returns the underlying OpenGL implementation type.

On platforms where the OpenGL implementation is not dynamically loaded, the return value is determined during compile time and never changes.

Note

A desktop OpenGL implementation may be capable of creating ES-compatible contexts too. Therefore in most cases it is more appropriate to check renderableType() or use the convenience function isOpenGLES() .

Note

This function requires that the QGuiApplication instance is already created.

resolveInterface(name, revision)
Parameters:
name – str

revision – int

Return type:
void

screen()
Return type:
QScreen

Returns the screen the context was created for.

See also

setScreen()

setFormat(format)
Parameters:
format – QSurfaceFormat

Sets the format the OpenGL context should be compatible with. You need to call create() before it takes effect.

When the format is not explicitly set via this function, the format returned by defaultFormat() will be used. This means that when having multiple contexts, individual calls to this function can be replaced by one single call to setDefaultFormat() before creating the first context.

See also

format()

setScreen(screen)
Parameters:
screen – QScreen

Sets the screen the OpenGL context should be valid for. You need to call create() before it takes effect.

See also

screen()

setShareContext(shareContext)
Parameters:
shareContext – QOpenGLContext

Makes this context share textures, shaders, and other OpenGL resources with shareContext. You need to call create() before it takes effect.

See also

shareContext()

shareContext()
Return type:
QOpenGLContext

Returns the share context this context was created with.

If the underlying platform was not able to support the requested sharing, this will return 0.

See also

setShareContext()

shareGroup()
Return type:
QOpenGLContextGroup

Returns the share group this context belongs to.

static supportsThreadedOpenGL()
Return type:
bool

Returns true if the platform supports OpenGL rendering outside the main (gui) thread.

The value is controlled by the platform plugin in use and may also depend on the graphics drivers.

surface()
Return type:
QSurface

Returns the surface the context has been made current with.

This is the surface passed as an argument to makeCurrent() .

swapBuffers(surface)
Parameters:
surface – QSurface

Swap the back and front buffers of surface.

Call this to finish a frame of OpenGL rendering, and make sure to call makeCurrent() again before issuing any further OpenGL commands, for example as part of a new frame.

Next
PySide6.QtGui.QOpenGLContextGroup
Previous
PySide6.QtGui.QOffscreenSurface
Copyright © 2025 The Qt Company Ltd. Documentation contributions included herein are the copyrights of their respective owners. The documentation provided herein is licensed under the terms of the GNU Free Documentation License version 1.3 (https://www.gnu.org/licenses/fdl.html) as published by the Free Software Foundation. Qt and respective logos are trademarks of The Qt Company Ltd. in Finland and/or other countries worldwide. All other trademarks are property of their respective owners.
Made with Sphinx and @pradyunsg's Furo
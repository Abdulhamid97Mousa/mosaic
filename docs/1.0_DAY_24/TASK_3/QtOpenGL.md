# https://doc.qt.io/qtforpython-6/PySide6/QtOpenGL/index.html#module-PySide6.QtOpenGL

PySide6.QtOpenGL
Detailed Description
OpenGL is a standard API for rendering 3D graphics. OpenGL only deals with 3D rendering and provides little or no support for GUI programming issues. The user interface for an OpenGL application must be created with another toolkit, such as XCB on the X platform, Microsoft Foundation Classes (MFC) under Windows, or Qt on both platforms.

Note

OpenGL is a trademark of Silicon Graphics, Inc. in the United States and other countries.

The Qt OpenGL module makes it easy to use OpenGL in Qt applications. To include the definitions of the module’s classes, use the following directive:

import PySide6.QtOpenGL
The Qt OpenGL module is implemented as a platform-independent wrapper around the platform-dependent GLX (version 1.3 or later), WGL, or AGL C APIs. Applications using the Qt OpenGL module can take advantage of the whole Qt API for non-OpenGL-specific GUI functionality.

The QtOpenGL module is available on Windows, X11 and Mac OS X. Qt for Embedded Linux and OpenGL supports OpenGL ES (OpenGL for Embedded Systems).

List of Classes
A

QAbstractOpenGLFunctions

B

Binder

O

QOpenGLBuffer

QOpenGLDebugLogger

QOpenGLDebugMessage

QOpenGLFramebufferObject

QOpenGLFramebufferObjectFormat

QOpenGLPaintDevice

QOpenGLPixelTransferOptions

QOpenGLShader

QOpenGLShaderProgram

QOpenGLTexture

QOpenGLTextureBlitter

QOpenGLTimeMonitor

QOpenGLTimerQuery

QOpenGLVersionFunctionsFactory

QOpenGLVersionProfile

QOpenGLVertexArrayObject

QOpenGLWindow


PySide6.QtQuick.QQuickFramebufferObject
class QQuickFramebufferObject
The QQuickFramebufferObject class is a convenience class for integrating OpenGL rendering using a framebuffer object (FBO) with Qt Quick.

Details
Warning

This class is only functional when Qt Quick is rendering via OpenGL. It is not compatible with other graphics APIs, such as Vulkan or Metal. It should be treated as a legacy class that is only present in order to enable Qt 5 applications to function without source compatibility breaks as long as they tie themselves to OpenGL.

On most platforms, the rendering will occur on a dedicated thread . For this reason, the QQuickFramebufferObject class enforces a strict separation between the item implementation and the FBO rendering. All item logic, such as properties and UI-related helper functions needed by QML should be located in a QQuickFramebufferObject class subclass. Everything that relates to rendering must be located in the Renderer class.

To avoid race conditions and read/write issues from two threads it is important that the renderer and the item never read or write shared variables. Communication between the item and the renderer should primarily happen via the synchronize() function. This function will be called on the render thread while the GUI thread is blocked.

Using queued connections or events for communication between item and renderer is also possible.

Both the Renderer and the FBO are memory managed internally.

To render into the FBO, the user should subclass the Renderer class and reimplement its render() function. The Renderer subclass is returned from createRenderer() .

The size of the FBO will by default adapt to the size of the item. If a fixed size is preferred, set textureFollowsItemSize to false and return a texture of your choosing from createFramebufferObject() .

Starting Qt 5.4, the QQuickFramebufferObject class is a texture provider and can be used directly in ShaderEffects and other classes that consume texture providers.

See also

Scene Graph and Rendering


Synopsis
Properties
mirrorVerticallyᅟ

textureFollowsItemSizeᅟ

Methods
def __init__()

def mirrorVertically()

def setMirrorVertically()

def setTextureFollowsItemSize()

def textureFollowsItemSize()

Virtual methods
def createRenderer()

Signals
def mirrorVerticallyChanged()

def textureFollowsItemSizeChanged()

Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

Note

Properties can be used directly when from __feature__ import true_property is used or via accessor functions otherwise.

property mirrorVerticallyᅟ: bool
This property controls if the size of the FBO’s contents should be mirrored vertically when drawing. This allows easy integration of third-party rendering code that does not follow the standard expectations.

The default value is false.

Access functions:
mirrorVertically()

setMirrorVertically()

Signal mirrorVerticallyChanged()

property textureFollowsItemSizeᅟ: bool
This property controls if the size of the FBO’s texture should follow the dimensions of the QQuickFramebufferObject item. When this property is false, the FBO will be created once the first time it is displayed. If it is set to true, the FBO will be recreated every time the dimensions of the item change.

The default value is true.

Access functions:
textureFollowsItemSize()

setTextureFollowsItemSize()

Signal textureFollowsItemSizeChanged()

__init__([parent=None])
Parameters:
parent – QQuickItem

Constructs a new QQuickFramebufferObject with parent parent.

abstract createRenderer()
Return type:
Renderer

Reimplement this function to create a renderer used to render into the FBO.

This function will be called on the rendering thread while the GUI thread is blocked.

mirrorVertically()
Return type:
bool

See also

setMirrorVertically()

Getter of property mirrorVerticallyᅟ .

mirrorVerticallyChanged(arg__1)
Parameters:
arg__1 – bool

Notification signal of property mirrorVerticallyᅟ .

setMirrorVertically(enable)
Parameters:
enable – bool

See also

mirrorVertically()

Setter of property mirrorVerticallyᅟ .

setTextureFollowsItemSize(follows)
Parameters:
follows – bool

See also

textureFollowsItemSize()

Setter of property textureFollowsItemSizeᅟ .

textureFollowsItemSize()
Return type:
bool

See also

setTextureFollowsItemSize()

Getter of property textureFollowsItemSizeᅟ .

textureFollowsItemSizeChanged(arg__1)
Parameters:
arg__1 – bool

Notification signal of property textureFollowsItemSizeᅟ .

class Renderer
Synopsis
Methods
def __init__()

def framebufferObject()

def invalidateFramebufferObject()

def update()

Virtual methods
def createFramebufferObject()

def render()

def synchronize()

Note

This documentation may contain snippets that were automatically translated from C++ to Python. We always welcome contributions to the snippet translation. If you see an issue with the translation, you can also let us know by creating a ticket on https:/bugreports.qt.io/projects/PYSIDE

Detailed Description
The Renderer class is used to implement the rendering logic of a QQuickFramebufferObject .

__init__()
Constructs a new renderer.

This function is called during the scene graph sync phase when the GUI thread is blocked.

createFramebufferObject(size)
Parameters:
size – QSize

Return type:
QOpenGLFramebufferObject

This function is called when a new FBO is needed. This happens on the initial frame. If textureFollowsItemSize is set to true, it is called again every time the dimensions of the item changes.

The returned FBO can have any attachment. If the QOpenGLFramebufferObjectFormat indicates that the FBO should be multisampled, the internal implementation of the Renderer will allocate a second FBO and blit the multisampled FBO into the FBO used to display the texture.

Note

Some hardware has issues with small FBO sizes. size takes that into account, so be cautious when overriding the size with a fixed size. A minimal size of 64x64 should always work.

Note

size takes the device pixel ratio into account, meaning that it is already multiplied by the correct scale factor. When moving the window containing the QQuickFramebufferObject item to a screen with different settings, the FBO is automatically recreated and this function is invoked with the correct size.

framebufferObject()
Return type:
QOpenGLFramebufferObject

Returns the framebuffer object currently being rendered to.

invalidateFramebufferObject()
Call this function during synchronize() to invalidate the current FBO. This will result in a new FBO being created with createFramebufferObject() .

abstract render()
This function is called when the FBO should be rendered into. The framebuffer is bound at this point and the glViewport has been set up to match the FBO size.

The FBO will be automatically unbound after the function returns.

Note

Do not assume that the OpenGL state is all set to the defaults when this function is invoked, or that it is maintained between calls. Both the Qt Quick renderer and the custom rendering code uses the same OpenGL context. This means that the state might have been modified by Quick before invoking this function.

Note

It is recommended to call resetOpenGLState() before returning. This resets OpenGL state used by the Qt Quick renderer and thus avoids interference from the state changes made by the rendering code in this function.

synchronize(item)
Parameters:
item – QQuickFramebufferObject

This function is called as a result of update() .

Use this function to update the renderer with changes that have occurred in the item. item is the item that instantiated this renderer. The function is called once before the FBO is created.

For instance, if the item has a color property which is controlled by QML, one should call update() and use synchronize() to copy the new color into the renderer so that it can be used to render the next frame.

This function is the only place when it is safe for the renderer and the item to read and write each others members.

update()
Call this function when the FBO should be rendered again.

This function can be called from render() to force the FBO to be rendered again before the next frame.

Note

This function should be used from inside the renderer. To update the item on the GUI thread, use update() .
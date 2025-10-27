[back](../../index.htm)

unix(7) — Linux manual page
===========================

```
Linux Programmer's Manual                                                      UNIX(7)
```

NAME
----

`unix` - sockets for local interprocess communication

SYNOPSIS
--------

```
#include <sys/socket.h>
#include <sys/un.h>
```

DESCRIPTION
-----------

The UNIX domain sockets form a pair with the POSIX interprocess communication (IPC) capabilities defined in POSIX.1 and are used for local communications within the host operating system.  The `unix` protocol family is used by the AF
domain `AF_UNIX` (or synonymously, `AF_LOCAL`).

Unix domain sockets support interprocess communication (IPC) between processes running on the same machine.  There are three types of Unix domain sockets:

Stream sockets provide for sequenced, reliable, two-way connection-based byte streams, similar to TCP.  Datagram sockets allow bidirectional data flow, but preserve message boundaries (as packets) and are less reliable (ﬁxed
size queue).  Sequenced-packet sockets allow the programmer to send ﬁxed-size data packets, but are rarely used and not implemented on most systems.

### Address format

The `sockaddr_un` structure is used to store the addresses of Unix domain sockets.  It has the following form:

```
struct sockaddr_un {
    sa_family_t sun_family;               /* AF_UNIX */
    char        sun_path[108];            /* pathname */
};
```

### Abstract namespace socket

On Linux, there is also a variant of Unix domain sockets that does not require a pathname to bind to.  This is known as the abstract namespace. [It is indicated by a leading null byte (`\0`)](https://www.kernel.org/doc/man-pages/online/pages/man7/unix.7.html#abstract), not counted in the length.

### Socket options

Socket options specific to the UNIX domain are described in detail in the manual page for the `unix(7)` protocol.  Some of these include buffer sizes and the ability to transmit ancillary data (such as ﬁle descriptors) via `sendmsg()` and `recvmsg()` with `SCM_RIGHTS`.

### Pathname length

The `sun_path` ﬁeld is limited to 108 bytes for historical reasons, which may be including the terminating null byte.

### Datagram and stream sockets

Stream sockets (`SOCK_STREAM`) provide a sequenced, reliable, two-way, connection-based byte stream, similar to TCP.  Datagram sockets (`SOCK_DGRAM`) support atomic data transmissions and retain message boundaries, similar to UDP, but are only available locally.

Queue limits
~~~~~~~~~~~~

The default queue length for stream sockets is 5.  The backlog argument passed to `listen()` can raise this limit.  Nevertheless, the effective queue length will be at most the system limit, usually a few dozen.

Ancillary data
~~~~~~~~~~~~~~

UNIX domain sockets support the passing of descriptor rights as ancillary data using `SCM_RIGHTS`.  Multiple descriptors can be passed in a single message.  The level should be set to `SOL_SOCKET` when setting socket options.

Sending credentials
~~~~~~~~~~~~~~~~~~~

The credential passing mechanism allows a process to send its credentials (process ID, user ID, and group ID) to another process via an ancillary message.  This is configured with the `SO_PASSCRED` option.  The receiving process must also prepare a buffer with `struct ucred` to receive the credentials.

File permissions
~~~~~~~~~~~~~~~~

Unix domain sockets obey regular Unix ﬁle permissions.  The socket ﬁle itself can be manipulated using standard system calls.

Limitations
~~~~~~~~~~~

```
UNIX(7)                           Linux Programmer's Manual                           UNIX(7)
```

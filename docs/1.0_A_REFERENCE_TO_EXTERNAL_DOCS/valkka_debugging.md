
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
* [Integrating with Qt and multiprocessing](qt_notes.html)
* [Multi-GPU systems](multi_gpu.html)
* [ValkkaFS](valkkafs.html)
* [Cloud Streaming](cloud.html)
* [OnVif & Discovery](onvif.html)
* [Common problems](pitfalls.html)
* Debugging
* [Repository Index](repos.html)
* [Licence & Copyright](license.html)
* [Authors](authors.html)
* [Knowledge Base](knowledge.html)

[Python Media Streaming Framework for Linux](index.html)

* Debugging
* [View page source](_sources/debugging.rst.txt)

---

# Debugging

*segfaults, memleaks, etc.*

LibValkka is rigorously “valgrinded” to remove any memory leaks at the cpp level. However, combining cpp and python (with swig) and throwing into the mix multithreading, multiprocessing and
sharing memory between processes, can (and will) give surprises.

**1. Check that you are not pulling frames from the same shared-memory channel using more than one client**

**2. Run Python + libValkka using gdb**

First, install python3 debugging symbols:

```
sudo apt-get install gdb python3-dbg
```

Then, create a custom build of libValkka with debug symbols enabled.

Finally, run your application’s entry point with:

```
gdb --args python3 python_program.py
run
```

See backtrace with

```
bt
```

If the trace point into `Objects/obmalloc.c`, then the cpp extensions have messed up python object reference counting. See also [here](https://stackoverflow.com/questions/26330621/python-segfaults-in-pyobject-malloc)

**3. Clear semaphores and shared memory every now and then by removing these files**

```
/dev/shm/*valkka*
```

**4. Follow python process memory consumption**

Use the [setproctitle python module](https://github.com/dvarrazzo/py-setproctitle) to name your python multiprocesses. This way you can find them easily using standard
linux monitoring tools, such as htop and smem.

Setting the name of the process should, of course, happen after the multiprocessing fork.

Install smem and htop:

```
sudo apt-get install smem htop
```

After that, run for example the script memwatch.bash in the aux/ directory. Or just launch htop. In htop, remember to go to setup => display options and enable “Hide userland process threads” to make
the output more readable.

Valkka-live, for example, names all multiprocesses adequately, so you can easily see if a process is
leaking memory.

**5. Prefer PyQt5 over PySide2**

You have the option of using PyQt5 instead of PySide2. The former is significantly more stable and handles the tricky
cpp Qt vs. Python reference counting correctly. Especially if you get that thing mention in (2), consider switching to PyQt5.

[Previous](pitfalls.html "Common problems")
[Next](repos.html "Repository Index")

---

© Copyright 2017-2020 Sampsa Riikonen.

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).

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
* Installing
  + [A. Install using PPA](#a-install-using-ppa)
  + [B. Install using releases](#b-install-using-releases)
  + [C. Compile yourself](#c-compile-yourself)
  + [Test your installation](#test-your-installation)
  + [Numpy](#numpy)
  + [Install the testsuite](#install-the-testsuite)
  + [GTK](#gtk)
  + [OpenCV](#opencv)
  + [Development version](#development-version)
* [The PyQt testsuite](testsuite.html)
* [Tutorial](tutorial.html)
* [Decoding](decoding.html)
* [Integrating with Qt and multiprocessing](qt_notes.html)
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

* Installing
* [View page source](_sources/requirements.rst.txt)

---

# Installing

The debian package includes the core library, its python bindings and some API level 2 python code. The python part is installed “globally” into */usr/lib/python3/dist-packages/*

Note

LibValkka comes precompiled and packaged for a certain ubuntu distribution version. This means that the compilation and it’s dependencies assume the default
python version of that distribution. Using custom-installed python versions, anacondas and whatnot might cause dependency problems.

## A. Install using PPA

*the preferred way*

For recent ubuntu distributions, the core library binary packages and python bindings are provided by a PPA repository. Subscribe to the PPA repo (do this only once) with:

```
sudo apt-add-repository ppa:sampsa-riikonen/valkka
```

Install with:

```
sudo apt-get update
sudo apt-get install valkka
```

When you need to update valkka, do:

```
sudo apt-get update
sudo apt-get install --only-upgrade valkka
```

## B. Install using releases

*if you don’t like PPAs*

You can download and install the required .deb packages “manually” from the
[releases page](https://github.com/elsampsa/valkka-core/releases)

```
sudo dpkg -i Valkka-*.deb
sudo apt-get install -fy
```

The last line pulls the dependencies.

Repeat the process when you need to update.

## C. Compile yourself

*the last resort*

If you’re not using a recent Ubuntu distro and need to build libValkka and it’s python bindings yourself,
please refer to the [valkka-core github page](https://github.com/elsampsa/valkka-core#compile-yourself).

## Test your installation

Test the installation with:

```
curl https://raw.githubusercontent.com/elsampsa/valkka-examples/master/quicktest.py -o quicktest.py
python3 quicktest.py
```

## Numpy

Valkka-core binaries has been compiled with the numpy version that comes with the corresponding Ubuntu distro, i.e. the numpy you would install with `sudo apt-get install python3-numpy`.

That version is automatically installed when you install valkka core with `sudo apt-get`, but it might be “shadowed” by your *locally* installed numpy.

If you get errors about numpy import, try removing your locally installed numpy (i.e. the version you installed with `pip install --user`).

## Install the testsuite

First, install some debian packages:

```
sudo apt-get install python3-pip git mesa-utils ffmpeg vlc
```

some of these will be used for benchmarking Valkka agains other programs.

The testsuite and tutorials use also imutils and PyQt5, so install a fresh version of them locally with pip:

```
pip3 install --user imutils PyQt5 PySide2 setproctitle
```

Here we have installed two flavors of the Qt python bindings, namely, [PyQt5](https://www.riverbankcomputing.com) and [PySide2](https://doc.qt.io/qtforpython/contents.html). They can be used in an identical manner. If you use PyQt5, be aware of its licensing terms.

Finally, for tutorial code and the PyQt test suite, download **valkka-examples** with:

```
git clone https://github.com/elsampsa/valkka-examples
```

Test the installation with:

```
cd valkka-examples
python3 quicktest.py
```

and you’re all set.

When updating the python examples (do this always after updating *valkka-core*), do the following:

```
git pull
python3 quicktest.py
```

This checks that **valkka-core** and **valkka-examples** have consistent versions.

In the case of a numerical python version mismatch error, you are not using the default numpy provided by your Ubuntu distribution (from the debian package *python3-numpy*). Remove the conflicting numpy installation with *pip3 uninstall* or setting up a virtualenv.

Next, try out the [PyQt test/demo](testsuite.html#testsuite) suite or learn to program with the [tutorial](tutorial.html#tutorial).

## GTK

If you wan’t to use [GTK](https://www.gtk.org/) as your graphical user interface, you must install the PyGObject python bindings, as instructed [here](https://pygobject.readthedocs.io/en/latest/getting_started.html), namely:

```
sudo apt-get install python-gi python-gi-cairo python3-gi python3-gi-cairo gir1.2-gtk-3.0
```

## OpenCV

Install with:

```
pip3 uninstall opencv-python
sudo pip3 uninstall opencv-python # just in case!
sudo apt-get install python3-opencv
```

The first one deinstall anything you may have installed with pip, while the second one installs the (good) opencv that
comes with your linux distro’s default python opencv installation.

## Development version

As described above, for the current stable version of *valkka-core*, just use the repository.

For the development version (with experimental and unstable features) you have to compile from source. You might need to do this also for
architectures other than x86.

Follow instructions in [here](https://github.com/elsampsa/valkka-core#compile-yourself).

[Previous](hardware.html "Supported hardware")
[Next](testsuite.html "The PyQt testsuite")

---

© Copyright 2017-2020 Sampsa Riikonen.

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).

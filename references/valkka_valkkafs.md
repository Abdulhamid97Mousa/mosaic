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
* [Integrating with Qt and multiprocessing](qt_notes.html)
* [Multi-GPU systems](multi_gpu.html)
* ValkkaFS
  + [VMS Architecture](#vms-architecture)
  + [Filesystem](#filesystem)
  + [Multiple Streams per File](#multiple-streams-per-file)
  + [Using an entire partition](#using-an-entire-partition)
* [Cloud Streaming](cloud.html)
* [OnVif & Discovery](onvif.html)
* [Common problems](pitfalls.html)
* [Debugging](debugging.html)
* [Repository Index](repos.html)
* [Licence & Copyright](license.html)
* [Authors](authors.html)
* [Knowledge Base](knowledge.html)

[Python Media Streaming Framework for Linux](index.html)

* ValkkaFS
* [View page source](_sources/valkkafs.rst.txt)

---

# ValkkaFS

## VMS Architecture

When creating a video management system capable of recording and playing simultaneously a large number of streams, here are *some* of the problems one encounters:

* Recorded video streams are playbacked in sync, i.e. their timestamps are matched together
* There can be large temporal chunks of video missing from any of the streams (i.e. the video stream is not continuous)
* Video is recorded simultaneously while it is being played, to/from the same file
* Only a finite amount of frames can be stored/buffered into memory for playback
* Book-keeping of the key-frames and rewinding from these key-frames to target time instants as per user’s requests

In order to solve this (surprisingly nasty) problem,
we have developed several objects and thread classes. Here is an overview of them:

* `core.ValkkaFS2` book-keeping of the frames
* `core.ValkkaFSReaderThread` reads frames from a file
* `core.ValkkaFSWriterThread` writes frames to a file
* `core.FileCacherThread` caches frames into memory and passes them down the filterchain
* Both `core.ValkkaFSReaderThread` and `core.ValkkaFSWriterThread` read/manipulate the book-keeping entity (`core.ValkkaFS2`) simultaneously

`core.ValkkaFS2`, `core.ValkkaFSReaderThread` and `core.ValkkaFSWriterThread` can be used for simple
dumping & reading streams to/from disk. Please see [the tutorial](lesson_11.html#valkkafs-tutorial).

For more complex solution, i.e. the mentioned simultaneous reading, writing & caching, the following filterchain (a) is used:

```
(core.ValkkaFSWriterThread) --> FILE --> (core.ValkkaFSReaderThread) --> (core.FileCacherThread)
                    slot-to-id      id-to-slot                                |
                                                                              |
                                               (DecoderThread) <--------------+
```

Typically, when recorded frames are played, the following takes place:

* Blocks of frames are requested from `core.ValkkaFSReaderThread`. From there they flow to `core.FileCacherThread`
* When play is requested from `core.FileCacherThread` it passes frames to the decoder at play speed

To make matter simpler for the API user the filterchain in (a) is further encapsulated into an `fs.FSGroup` object.

Several `FSGroup` objects are further encapsulated into a `fs.ValkkaFSManager` object, the final hierarchical object encapsulation
looking like this:

```
fs.ValkkaFSManager
    fs.FSGroup
        fs.ValkkaSingleFS
            core.ValkkaFS2
        core.ValkkaFSWriterThread
        core.ValkkaFSReaderThread
        core.FileCacherThread
    fs.FSGroup
        fs.ValkkaSingleFS
            core.ValkkaFS2
        core.ValkkaFS2
        core.ValkkaFSWriterThread
        ...
```

`ValkkaFSManager` being the “end-point”, from where a user can request synchronized playing and seeking for a number of streams. `ValkkaFSManager` would be typically connected
 to a GUI component for interactive playback.

Please refer to the [PyQt testsuite](testsuite.html#testsuite) on how to use `FSGroup` and `ValkkaFSManager`.

## Filesystem

ValkkaSingleFS is a simple filesystem for storing streaming media data. Video frames (H264) are organized in “blocks” and written into a “dumpfile”.
The dumpfile can be a regular file or a dedicated disk partition.

Dumpfile is *pre-reserved*, which makes the life easier for the underlying filesystem and avoids fragmentation (in contrast to creating a huge amount of small, timestamped video segment files).

The size of a single block (S) and the number of blocks (N) are predefined by the user. The total disk space for a recording is then N*S bytes.

Once the last block is written, writing is “wrapped” and resumed from block number 1. This way the oldest recorded stream is overwritten automatically.

Per each ValkkaFS, a directory is created with the following files:

```
directory/
    blockfile       # book-keeping of the frames
    dumpfile        # recorded H264 stream
    valkkafs.json   # metadata
```

`blockfile` is simple binary file that encapsulates a table with N (number of blocks) rows and two columns. Each column represents a millisecond timestamp:

```
mstime1     mstime2
...         ...
```

where `mstime1` indicates the first *key-frame* available in a block, while `mstime2` indicates the last frame available in that block.

`valkkafs.json` saves data about the current written block, number of blocks and the block size.

For efficient recording and playback with ValkkaFS (or with *any* VMS system), consider this:

* For efficient seeking, set your camera to emit **one key-frame per second** (or even two)
* Be aware of the bitrate of your camera and adjust the blocksize in ValkkaFS to that: ideally you’d want **1-2 key frames per block**

Consult [the tutorial](lesson_11.html#valkkafs-tutorial) for more details.

## Multiple Streams per File

You can also dump multiple streams into a single `ValkkaFS`. The variant for this is `valkka.fs.ValkkaMultiFS`.

This requires that all cameras have the same bitrate and key-frame interval!

The advantage of this approach is, that all frames from all your cameras are streamed continuously into the same (large) file or a dedicated block device,
minimizing the wear and tear on your device if you are using a hdd.

The architecture is identical to `ValkkaSingleFS`, with a very small modification to the `blockfile` format: `mstime1` presents
now the *last* key-frame among all keyframes of all the streams.

*WARNING: writing multiple streams to the same file / block device is at very experimental stage and not well tested*

## Using an entire partition

*WARNING: this makes sense only if you are using ValkkaMultiFS, i.e. streaming several cameras into a same ValkkaFS*

An entire hard-drive/partition can be dedicated to ValkkaFS. In the following example, we assume that your external hard-disk appears under */dev/sdb*

To grant access to a linux user to read and write block devices directly, use the following command:

```
sudo usermod -a -G disk username
```

After that you still need to logout and login again.

Now you can verify that block devices can be read and written as regular files. Try this command:

```
head -n 10 /dev/sdb
```

to read the first ten bytes of that external hard-drive.

ValkkaFS uses devices with **GPT partition tables**, having **Linux swap partitions**, located **on block devices**.

Why such a scheme? We’ll be writing over that partition, so we just wan’t to be sure it’s not a normal user filesystem. :)

The next thing we need, is to create a Linux swap partition on that external (or internal) hard disk. The recommended tool for this is *gparted*.

Start gparted with:

```
sudo gparted /dev/sdb
```

Once in gparted, choose *device* => *create partition table*. Choose *gpt* partition table and press *apply*. Next choose *partition*, and there, choose *linux swap*.

Let’s see how it worked out, so type

```
sudo fdisk -l
```

You should get something like this:

```
Device     Start   End        Sectors   Size    Type
/dev/sdb1  2048    976773134  976771087 465,8G  Linux swap
```

To get the exact size in bytes, type:

```
blockdev --getsize64 /dev/sdb1
```

So, in this case we’d be dedicating an external USB drive of 465 GB for recording streaming video.

To identify disks, Valkka uses uuid partition identification. The uuid can be found with

```
blkid /dev/sdb1
```

Suppose you get:

```
/dev/sdb1: UUID="db572185-2ac1-4ef5-b8af-c2763e639a67" TYPE="swap" PARTUUID="37c591e3-b33b-4548-a1eb-81add9da8a58"
```

Then “37c591e3-b33b-4548-a1eb-81add9da8a58” is what you are looking for.

In this example case, you would instantiate the ValkkaFS like this:

```
valkkafs = ValkkaMultiFS.newFromDirectory(
    dirname="/home/sampsa/tmp/testvalkkafs",
    partition_uuid="37c591e3-b33b-4548-a1eb-81add9da8a58",
    blocksize=YOUR_BLOCKSIZE,
    device_size=1024*1024*1024*465) # 465 GB
```

[Previous](multi_gpu.html "Multi-GPU systems")
[Next](cloud.html "Cloud Streaming")

---

© Copyright 2017-2020 Sampsa Riikonen.

Built with [Sphinx](https://www.sphinx-doc.org/) using a
[theme](https://github.com/readthedocs/sphinx_rtd_theme)
provided by [Read the Docs](https://readthedocs.org).

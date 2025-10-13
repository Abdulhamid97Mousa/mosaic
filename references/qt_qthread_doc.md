[Back to Qt.io](https://www.qt.io/)

[Contact Us](https://www.qt.io/contact-us/)
[Blog](https://blog.qt.io/)
[Download Qt](https://www.qt.io/download/)

[![Qt documentation](/style/qt-logo-documentation.png)](/)

* + English
* [ ]
* [Archives](/archives)
* [Snapshots](https://doc-snapshots.qt.io/)

Search

* [Qt 6.10](index.html)
* [Qt Core](qtcore-index.html)
* [C++ Classes](qtcore-module.html)
* QThread

# QThread Class

The QThread class provides a platform-independent way to manage threads. [More...](#details)

|  |  |
| --- | --- |
| Header: | `#include <QThread>` |
| CMake: | `find_package(Qt6 REQUIRED COMPONENTS Core)`  `target_link_libraries(mytarget PRIVATE Qt6::Core)` |
| qmake: | `QT += core` |
| Inherits: | [QObject](qobject.html) |

* [List of all members, including inherited members](qthread-members.html)
* QThread is part of [Threading Classes](thread.html).

## Public Types

|  |  |
| --- | --- |
| enum | **[Priority](qthread.html#Priority-enum)** { IdlePriority, LowestPriority, LowPriority, NormalPriority, HighPriority, …, InheritPriority } |
| `(since 6.9)` enum class | **[QualityOfService](qthread.html#QualityOfService-enum)** { Auto, High, Eco } |

## Public Functions

|  |  |
| --- | --- |
|  | **[QThread](qthread.html#QThread)**(QObject **parent* = nullptr) |
| virtual | **[~QThread](qthread.html#dtor.QThread)**() |
| QAbstractEventDispatcher * | **[eventDispatcher](qthread.html#eventDispatcher)**() const |
| `(since 6.8)` bool | **[isCurrentThread](qthread.html#isCurrentThread)**() const |
| bool | **[isFinished](qthread.html#isFinished)**() const |
| bool | **[isInterruptionRequested](qthread.html#isInterruptionRequested)**() const |
| bool | **[isRunning](qthread.html#isRunning)**() const |
| int | **[loopLevel](qthread.html#loopLevel)**() const |
| QThread::Priority | **[priority](qthread.html#priority)**() const |
| void | **[requestInterruption](qthread.html#requestInterruption)**() |
| `(since 6.9)` QThread::QualityOfService | **[serviceLevel](qthread.html#serviceLevel)**() const |
| void | **[setEventDispatcher](qthread.html#setEventDispatcher)**(QAbstractEventDispatcher **eventDispatcher*) |
| void | **[setPriority](qthread.html#setPriority)**(QThread::Priority *priority*) |
| `(since 6.9)` void | **[setServiceLevel](qthread.html#setServiceLevel)**(QThread::QualityOfService *serviceLevel*) |
| void | **[setStackSize](qthread.html#setStackSize)**(uint *stackSize*) |
| uint | **[stackSize](qthread.html#stackSize)**() const |
| bool | **[wait](qthread.html#wait)**(QDeadlineTimer *deadline* = QDeadlineTimer(QDeadlineTimer::Forever)) |
| bool | **[wait](qthread.html#wait-1)**(unsigned long *time*) |

## Reimplemented Public Functions

|  |  |
| --- | --- |
| virtual bool | **[event](qthread.html#event)**(QEvent **event*) override |

## Public Slots

|  |  |
| --- | --- |
| void | **[exit](qthread.html#exit)**(int *returnCode* = 0) |
| void | **[quit](qthread.html#quit)**() |
| void | **[start](qthread.html#start)**(QThread::Priority *priority* = InheritPriority) |
| void | **[terminate](qthread.html#terminate)**() |

## Signals

|  |  |
| --- | --- |
| void | **[finished](qthread.html#finished)**() |
| void | **[started](qthread.html#started)**() |

## Static Public Members

|  |  |
| --- | --- |
| QThread * | **[create](qthread.html#create)**(Function &&*f*, Args &&... *args*) |
| QThread * | **[currentThread](qthread.html#currentThread)**() |
| Qt::HANDLE | **[currentThreadId](qthread.html#currentThreadId)**() |
| int | **[idealThreadCount](qthread.html#idealThreadCount)**() |
| `(since 6.8)` bool | **[isMainThread](qthread.html#isMainThread)**() |
| void | **[msleep](qthread.html#msleep)**(unsigned long *msecs*) |
| `(since 6.6)` void | **[sleep](qthread.html#sleep)**(std::chrono::nanoseconds *nsecs*) |
| void | **[sleep](qthread.html#sleep-1)**(unsigned long *secs*) |
| void | **[usleep](qthread.html#usleep)**(unsigned long *usecs*) |
| void | **[yieldCurrentThread](qthread.html#yieldCurrentThread)**() |

## Protected Functions

|  |  |
| --- | --- |
| int | **[exec](qthread.html#exec)**() |
| virtual void | **[run](qthread.html#run)**() |

## Static Protected Members

|  |  |
| --- | --- |
| void | **[setTerminationEnabled](qthread.html#setTerminationEnabled)**(bool *enabled* = true) |

## Detailed Description

A QThread object manages one thread of control within the program. QThreads begin executing in [run](qthread.html#run)(). By default, [run](qthread.html#run)() starts the event loop by calling [exec](qthread.html#exec)() and runs a Qt event loop inside the thread.

You can use worker objects by moving them to the thread using [QObject::moveToThread](qobject.html#moveToThread)().

```
class Worker : public QObject
{
    Q_OBJECT

public slots:
    void doWork(const QString &parameter) {
        QString result;
        /* ... here is the expensive or blocking operation ... */
        emit resultReady(result);
    }

signals:
    void resultReady(const QString &result);
};

class Controller : public QObject
{
    Q_OBJECT
    QThread workerThread;
public:
    Controller() {
        Worker *worker = new Worker;
        worker->moveToThread(&workerThread);
        connect(&workerThread, &QThread::finished, worker, &QObject::deleteLater);
        connect(this, &Controller::operate, worker, &Worker::doWork);
        connect(worker, &Worker::resultReady, this, &Controller::handleResults);
        workerThread.start();
    }
    ~Controller() {
        workerThread.quit();
        workerThread.wait();
    }
public slots:
    void handleResults(const QString &);
signals:
    void operate(const QString &);
};
```

The code inside the Worker's slot would then execute in a separate thread. However, you are free to connect the Worker's slots to any signal, from any object, in any thread. It is safe to connect signals and slots across different threads, thanks to a mechanism called [queued connections](qt.html#ConnectionType-enum).

Another way to make code run in a separate thread, is to subclass QThread and reimplement [run](qthread.html#run)(). For example:

```
class WorkerThread : public QThread
{
    Q_OBJECT
public:
    explicit WorkerThread(QObject *parent = nullptr) : QThread(parent) { }
protected:
    void run() override {
        QString result;
        /* ... here is the expensive or blocking operation ... */
        emit resultReady(result);
    }
signals:
    void resultReady(const QString &s);
};

void MyObject::startWorkInAThread()
{
    WorkerThread *workerThread = new WorkerThread(this);
    connect(workerThread, &WorkerThread::resultReady, this, &MyObject::handleResults);
    connect(workerThread, &WorkerThread::finished, workerThread, &QObject::deleteLater);
    workerThread->start();
}
```

In that example, the thread will exit after the run function has returned. There will not be any event loop running in the thread unless you call [exec](qthread.html#exec)().

It is important to remember that a QThread instance [lives in](qobject.html#thread-affinity) the old thread that instantiated it, not in the new thread that calls [run](qthread.html#run)(). This means that all of QThread's queued slots and [invoked methods](qmetaobject.html#invokeMethod) will execute in the old thread. Thus, a developer who wishes to invoke slots in the new thread must use the worker-object approach; new slots should not be implemented directly into a subclassed QThread.

Unlike queued slots or invoked methods, methods called directly on the QThread object will execute in the thread that calls the method. When subclassing QThread, keep in mind that the constructor executes in the old thread while [run](qthread.html#run)() executes in the new thread. If a member variable is accessed from both functions, then the variable is accessed from two different threads. Check that it is safe to do so.

**Note:** Care must be taken when interacting with objects across different threads. As a general rule, functions can only be called from the thread that created the QThread object itself (e.g. [setPriority](qthread.html#setPriority)()), unless the documentation says otherwise. See [Synchronizing Threads](threads-synchronizing.html) for details.

### Managing Threads

QThread will notify you via a signal when the thread is [started](qthread.html#started)() and [finished](qthread.html#finished)(), or you can use [isFinished](qthread.html#isFinished)() and [isRunning](qthread.html#isRunning)() to query the state of the thread.

You can stop the thread by calling [exit](qthread.html#exit)() or [quit](qthread.html#quit)(). In extreme cases, you may want to forcibly [terminate](qthread.html#terminate)() an executing thread. However, doing so is dangerous and discouraged. Please read the documentation for [terminate](qthread.html#terminate)() and [setTerminationEnabled](qthread.html#setTerminationEnabled)() for detailed information.

You often want to deallocate objects that live in a thread when a thread ends. To do this, connect the [finished](qthread.html#finished)() signal to [QObject::deleteLater](qobject.html#deleteLater)().

Use [wait](qthread.html#wait)() to block the calling thread, until the other thread has finished execution (or until a specified time has passed).

QThread also provides static, platform independent sleep functions: [sleep](qthread.html#sleep)(), [msleep](qthread.html#msleep)(), and [usleep](qthread.html#usleep)() allow full second, millisecond, and microsecond resolution respectively.

**Note:** [wait](qthread.html#wait)() and the [sleep](qthread.html#sleep)() functions should be unnecessary in general, since Qt is an event-driven framework. Instead of [wait](qthread.html#wait)(), consider listening for the [finished](qthread.html#finished)() signal. Instead of the [sleep](qthread.html#sleep)() functions, consider using [QChronoTimer](qchronotimer.html).

The static functions [currentThreadId](qthread.html#currentThreadId)() and [currentThread](qthread.html#currentThread)() return identifiers for the currently executing thread. The former returns a platform specific ID for the thread; the latter returns a QThread pointer.

To choose the name that your thread will be given (as identified by the command `ps -L` on Linux, for example), you can call [setObjectName](qobject.html#setObjectName)() before starting the thread. If you don't call [setObjectName](qobject.html#setObjectName)(), the name given to your thread will be the class name of the runtime type of your thread object (for example, `"RenderThread"` in the case of the [Mandelbrot](qtcore-threads-mandelbrot-example.html) example, as that is the name of the QThread subclass). Note that this is currently not available with release builds on Windows.

**See also** [Multi-threading in Qt](threads.html), [QThreadStorage](qthreadstorage.html), [Synchronizing Threads](threads-synchronizing.html), [Mandelbrot](qtcore-threads-mandelbrot-example.html), [Producer and Consumer using Semaphores](qtcore-threads-semaphores-example.html), and [Producer and Consumer using Wait Conditions](qtcore-threads-waitconditions-example.html).

## Member Type Documentation

### enum QThread::Priority

This enum type indicates how the operating system should schedule newly created threads.

| Constant | Value | Description |
| --- | --- | --- |
| `QThread::IdlePriority` | `0` | scheduled only when no other threads are running. |
| `QThread::LowestPriority` | `1` | scheduled less often than LowPriority. |
| `QThread::LowPriority` | `2` | scheduled less often than NormalPriority. |
| `QThread::NormalPriority` | `3` | the default priority of the operating system. |
| `QThread::HighPriority` | `4` | scheduled more often than NormalPriority. |
| `QThread::HighestPriority` | `5` | scheduled more often than HighPriority. |
| `QThread::TimeCriticalPriority` | `6` | scheduled as often as possible. |
| `QThread::InheritPriority` | `7` | use the same priority as the creating thread. This is the default. |

### `[since 6.9]` enum class QThread::QualityOfService

This enum describes the quality of service level of a thread, and provides the scheduler with information about the kind of work that the thread performs. On platforms with different CPU profiles, or with the ability to clock certain cores of a CPU down, this allows the scheduler to select or configure a CPU core with suitable performance and energy characteristics for the thread.

| Constant | Value | Description |
| --- | --- | --- |
| `QThread::QualityOfService::Auto` | `0` | The default value, leaving it to the scheduler to decide which CPU core to run the thread on. |
| `QThread::QualityOfService::High` | `1` | The scheduler should run this thread to a high-performance CPU core. |
| `QThread::QualityOfService::Eco` | `2` | The scheduler should run this thread to an energy-efficient CPU core. |

This enum was introduced in Qt 6.9.

**See also** [Priority](qthread.html#Priority-enum), [serviceLevel](qthread.html#serviceLevel)(), and [QThreadPool::serviceLevel](qthreadpool.html#serviceLevel)().

## Member Function Documentation

### `[explicit]` QThread::QThread([QObject](qobject.html) **parent* = nullptr)

Constructs a new QThread to manage a new thread. The *parent* takes ownership of the QThread. The thread does not begin executing until [start](qthread.html#start)() is called.

**See also** [start](qthread.html#start)().

### `[virtual noexcept]` QThread::~QThread()

Destroys the [QThread](qthread.html).

Note that deleting a [QThread](qthread.html) object will not stop the execution of the thread it manages. Deleting a running [QThread](qthread.html) (i.e. [isFinished](qthread.html#isFinished)() returns `false`) will result in a program crash. Wait for the [finished](qthread.html#finished)() signal before deleting the [QThread](qthread.html).

Since Qt 6.3, it is allowed to delete a [QThread](qthread.html) instance created by a call to [QThread::create](qthread.html#create)() even if the corresponding thread is still running. In such a case, Qt will post an interruption request to that thread (via [requestInterruption](qthread.html#requestInterruption)()); will ask the thread's event loop (if any) to quit (via [quit](qthread.html#quit)()); and will block until the thread has finished.

**See also** [create](qthread.html#create)(), [isInterruptionRequested](qthread.html#isInterruptionRequested)(), [exec](qthread.html#exec)(), and [quit](qthread.html#quit)().

...

© 2025 The Qt Company Ltd.
Documentation contributions included herein are the copyrights of
their respective owners. The documentation provided herein is licensed under the terms of the [GNU Free Documentation License version 1.3](http://www.gnu.org/licenses/fdl.html) as published by the Free Software Foundation. Qt and respective logos are  [trademarks](https://doc.qt.io/qt/trademarks.html) of The Qt Company Ltd. in Finland and/or other countries
worldwide. All other trademarks are property of their respective owners.

###### **Contents**

* [Public Types](#public-types)
* [Public Functions](#public-functions)
* [Reimplemented Public Functions](#reimplemented-public-functions)
* [Public Slots](#public-slots)
* [Signals](#signals)
* [Static Public Members](#static-public-members)
* [Protected Functions](#protected-functions)
* [Static Protected Members](#static-protected-members)
* [Detailed Description](#details)
* [Managing Threads](#managing-threads)

[![](/images/qtgroup.svg)](https://www.qt.io/?hsLang=en)

[Contact Us](https://www.qt.io/contact-us?hsLang=en)

* Qt Group
  + [Our Story](https://www.qt.io/group)
  + [Brand](https://www.qt.io/brand)
  + [News](https://www.qt.io/newsroom)
  + [Careers](https://www.qt.io/careers)
  + [Investors](https://www.qt.io/investors)
  + [Qt Products](https://www.qt.io/product)
  + [Quality Assurance Products](https://www.qt.io/product/quality-assurance)
* Licensing
  + [License Agreement](https://www.qt.io/terms-conditions)
  + [Open Source](https://www.qt.io/licensing/open-source-lgpl-obligations)
  + [Plans and pricing](https://www.qt.io/pricing)
  + [Download](https://www.qt.io/download)
  + [FAQ](https://www.qt.io/faq/overview)
* Learn Qt
  + [For Learners](https://www.qt.io/academy)
  + [For Students and Teachers](https://www.qt.io/qt-educational-license)
  + [Qt Documentation](https://doc.qt.io/)
  + [Qt Forum](https://forum.qt.io/)
* Support & Services
  + [Professional Services](https://www.qt.io/qt-professional-services)
  + [Customer Success](https://www.qt.io/customer-success)
  + [Support Services](https://www.qt.io/qt-support/)
  + [Partners](https://www.qt.io/contact-us/partners)
  + [Qt World](https://www.qt.io/qt-world)

* © 2025 The Qt Company
* Feedback

Qt Group includes The Qt Company Oy and its global subsidiaries and affiliates.

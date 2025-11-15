The QML Reference
A multi-paradigm language for application development

QML is a multi-paradigm language for creating highly dynamic applications. With QML, application building blocks such as UI components are declared and various properties set to define the application behavior. Application behavior can be further scripted through JavaScript, which is a subset of the language. In addition, QML heavily uses Qt, which allows types and other Qt features to be accessible directly from QML applications.

This reference guide describes the features of the QML language. Many of the QML types in the guide originate from the Qt Qml or Qt Quick modules.

QML Syntax Basics

Import Statements

Object Declarations

Child Objects

Comments

QML Object Attributes

id :ref:` <QML-Object-Attributes>` Attribute

Property Attributes

Signal Attributes

Method Attributes

Attached Properties and Attached Signal Handlers

Enumeration Attributes

Property Binding

Signal and Handler Event System

Integrating QML and JavaScript

Using JavaScript Expressions with QML

Dynamic QML Object Creation from JavaScript

Defining JavaScript Resources In QML

Importing JavaScript Resources In QML

JavaScript Host Environment

Configuring the JavaScript engine

The QML Type System

QML Value Types

JavaScript Types

QML Object Types

Defining Object Types from QML

Defining Object Types from C++

QML Sequence Types

QML Namespaces

QML Modules

Specifying A QML Module

Supported QML Module Types

Identified Modules

Legacy Modules

Providing Types and Functionality in a C++ Plugin

QML Documents

Structure of a QML Document

Syntax of the QML Language

Defining Object Types through QML Documents

Defining an Object Type with a QML File

Accessible Attributes of Custom Types

Resource Loading and Network Transparency

Scope and Naming Resolution





## PySide6.QtQml

## https://doc.qt.io/qtforpython-6/PySide6/QtQml/index.html


Detailed Description
The Qt Qml module implements the QML language and offers APIs to register types for it.

The Qt Qml module provides a framework for developing applications and libraries with the The QML Reference . It defines and implements the language and engine infrastructure, and provides an API to enable application developers to register custom QML types and modules and integrate QML code with JavaScript and Python. The Qt Qml module provides both a QML API a Python API.

Using the Module
To include the definitions of modules classes, use the following directive:

import PySide6.QtQml
Registering QML Types and QML Modules
See Python-QML integration.

Tweaking the engine
There are a number of knobs you can turn to customize the behavior of the QML engine. The page on Configuring the JavaScript Engine lists the environment variables you may use to this effect. The description of The QML Disk Cache describes the options related to how your QML components are compiled and loaded.

List of QML types
QtQml QML Types

List of Classes
J

QJSEngine

QJSManagedValue

QJSPrimitiveValue

QJSValue

QJSValueIterator

L

ListProperty

P

PropertyPair

QPyQmlParserStatus

QPyQmlPropertyValueSource

Q

QQmlAbstractUrlInterceptor

QQmlApplicationEngine

QQmlComponent

QQmlContext

QQmlDebuggingEnabler

QQmlEngine

QQmlError

QQmlExpression

QQmlExtensionInterface

QQmlExtensionPlugin

QQmlFile

QQmlFileSelector

QQmlImageProviderBase

QQmlIncubationController

QQmlIncubator

QQmlListReference

QQmlNetworkAccessManagerFactory

QQmlParserStatus

QQmlProperty

QQmlPropertyMap

QQmlPropertyValueSource

QQmlScriptString

QQmlTypesExtensionInterface

List of Decorators
Q

@QmlAnonymous

@QmlAttached

@QmlElement

@QmlExtended

@QmlForeign

@QmlNamedElement

@QmlSingleton

@QmlUncreatable

List of Functions
Q

qjsEngine()

qmlClearTypeRegistrations()

qmlContext()

qmlEngine()

qmlProtectModule()

qmlRegisterModule()

qmlRegisterSingletonType()

qmlRegisterType()

qmlRegisterUncreatableMetaObject()

qmlTypeId()

qmlAttachedPropertiesObject()

qmlRegisterSingletonInstance()

qmlRegisterUncreatableType()

List of Enumerations
M

QML_HAS_ATTACHED_PROPERTIES

Q

QQmlModuleImportSpecialVersions




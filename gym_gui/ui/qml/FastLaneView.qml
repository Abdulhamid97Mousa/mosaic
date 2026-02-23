import QtQuick 2.15

import GymGui 1.0

Item {
    id: root
    objectName: "fastlaneRoot"
    width: 640
    height: 480
    property alias metricsLabel: metricsText.text
    property string hudText: "" 
    signal requestFrame()
    signal frameDropped()

    FastLaneItem {
        id: fastlaneCanvas
        objectName: "fastlaneCanvas"
        anchors.fill: parent
    }

    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 12
        color: "#40000000"
        radius: 6
        border.width: 0
        Column {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 4
            Text {
                id: metricsText
                text: root.hudText
                color: "white"
                font.pixelSize: 14
                wrapMode: Text.Wrap
            }
        }
    }
}

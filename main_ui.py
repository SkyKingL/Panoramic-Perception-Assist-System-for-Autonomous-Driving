# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1039, 889)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(6, 0, 1031, 841))
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.out_video = QLabel(self.tab)
        self.out_video.setObjectName(u"out_video")
        self.out_video.setGeometry(QRect(1, 40, 1021, 611))
        self.groupBox_6 = QGroupBox(self.tab)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setGeometry(QRect(170, 700, 691, 101))
        self.pause = QPushButton(self.groupBox_6)
        self.pause.setObjectName(u"pause")
        self.pause.setGeometry(QRect(352, 19, 93, 28))
        self.Init = QPushButton(self.groupBox_6)
        self.Init.setObjectName(u"Init")
        self.Init.setGeometry(QRect(242, 20, 93, 28))
        self.select_img = QPushButton(self.groupBox_6)
        self.select_img.setObjectName(u"select_img")
        self.select_img.setGeometry(QRect(170, 60, 93, 28))
        self.select_video = QPushButton(self.groupBox_6)
        self.select_video.setObjectName(u"select_video")
        self.select_video.setGeometry(QRect(299, 60, 93, 28))
        self.exit = QPushButton(self.groupBox_6)
        self.exit.setObjectName(u"exit")
        self.exit.setGeometry(QRect(430, 60, 93, 28))
        self.label_6 = QLabel(self.tab)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(0, 0, 72, 15))
        self.fps_label = QLabel(self.tab)
        self.fps_label.setObjectName(u"fps_label")
        self.fps_label.setGeometry(QRect(10, 10, 72, 15))
        self.msg = QLabel(self.tab)
        self.msg.setObjectName(u"msg")
        self.msg.setGeometry(QRect(0, 670, 1021, 20))
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.groupBox_7 = QGroupBox(self.tab_2)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setGeometry(QRect(42, 69, 121, 31))
        self.label = QLabel(self.groupBox_7)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(13, 10, 101, 16))
        self.groupBox_8 = QGroupBox(self.tab_2)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setGeometry(QRect(42, 137, 121, 31))
        self.label_2 = QLabel(self.groupBox_8)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(13, 10, 101, 16))
        self.imgpath = QLineEdit(self.tab_2)
        self.imgpath.setObjectName(u"imgpath")
        self.imgpath.setGeometry(QRect(202, 79, 561, 21))
        self.videopath = QLineEdit(self.tab_2)
        self.videopath.setObjectName(u"videopath")
        self.videopath.setGeometry(QRect(202, 142, 561, 21))
        self.imgsave = QPushButton(self.tab_2)
        self.imgsave.setObjectName(u"imgsave")
        self.imgsave.setGeometry(QRect(776, 75, 93, 28))
        self.videosave = QPushButton(self.tab_2)
        self.videosave.setObjectName(u"videosave")
        self.videosave.setGeometry(QRect(777, 138, 93, 28))
        self.groupBox_9 = QGroupBox(self.tab_2)
        self.groupBox_9.setObjectName(u"groupBox_9")
        self.groupBox_9.setGeometry(QRect(30, 380, 961, 231))
        self.layoutWidget = QWidget(self.groupBox_9)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(150, 90, 721, 28))
        self.horizontalLayout_7 = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_7.setSpacing(5)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.nms_SpinBoX = QDoubleSpinBox(self.layoutWidget)
        self.nms_SpinBoX.setObjectName(u"nms_SpinBoX")
        self.nms_SpinBoX.setMinimumSize(QSize(50, 0))
        self.nms_SpinBoX.setMaximumSize(QSize(50, 16777215))
        self.nms_SpinBoX.setStyleSheet(u"")
        self.nms_SpinBoX.setMaximum(1.000000000000000)
        self.nms_SpinBoX.setSingleStep(0.010000000000000)
        self.nms_SpinBoX.setValue(0.450000000000000)

        self.horizontalLayout_7.addWidget(self.nms_SpinBoX)

        self.nms_Slider = QSlider(self.layoutWidget)
        self.nms_Slider.setObjectName(u"nms_Slider")
        self.nms_Slider.setStyleSheet(u"QSlider{\n"
"border-color: #bcbcbc;\n"
"color:#d9d9d9;\n"
"}\n"
"QSlider::groove:horizontal {                                \n"
"     border: 1px solid #999999;                             \n"
"     height: 3px;                                           \n"
"    margin: 0px 0;                                         \n"
"     left: 5px; right: 5px; \n"
" }\n"
"QSlider::handle:horizontal {                               \n"
"     border: 0px ; \n"
"     border-image: url(:/img/icon/\u5706.png);\n"
"	 width:15px;\n"
"     margin: -7px -7px -7px -7px;                  \n"
"} \n"
"QSlider::add-page:horizontal{\n"
"background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
"\n"
"}\n"
"QSlider::sub-page:horizontal{                               \n"
" background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
"}")
        self.nms_Slider.setMaximum(100)
        self.nms_Slider.setValue(45)
        self.nms_Slider.setOrientation(Qt.Horizontal)

        self.horizontalLayout_7.addWidget(self.nms_Slider)

        self.label_4 = QLabel(self.groupBox_9)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(30, 95, 111, 16))
        self.layoutWidget_2 = QWidget(self.groupBox_9)
        self.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.layoutWidget_2.setGeometry(QRect(150, 160, 721, 28))
        self.horizontalLayout_8 = QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_8.setSpacing(5)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.conf_SpinBox = QDoubleSpinBox(self.layoutWidget_2)
        self.conf_SpinBox.setObjectName(u"conf_SpinBox")
        self.conf_SpinBox.setMinimumSize(QSize(50, 0))
        self.conf_SpinBox.setMaximumSize(QSize(50, 16777215))
        self.conf_SpinBox.setFocusPolicy(Qt.ClickFocus)
        self.conf_SpinBox.setStyleSheet(u"")
        self.conf_SpinBox.setMaximum(1.000000000000000)
        self.conf_SpinBox.setSingleStep(0.010000000000000)
        self.conf_SpinBox.setValue(0.250000000000000)

        self.horizontalLayout_8.addWidget(self.conf_SpinBox)

        self.conf_Slider = QSlider(self.layoutWidget_2)
        self.conf_Slider.setObjectName(u"conf_Slider")
        self.conf_Slider.setStyleSheet(u"QSlider{\n"
"border-color: #bcbcbc;\n"
"color:#d9d9d9;\n"
"}\n"
"QSlider::groove:horizontal {                                \n"
"     border: 1px solid #999999;                             \n"
"     height: 3px;                                           \n"
"    margin: 0px 0;                                         \n"
"     left: 5px; right: 5px; \n"
" }\n"
"QSlider::handle:horizontal {                               \n"
"     border: 0px ; \n"
"     border-image: url(:/img/icon/\u5706.png);\n"
"	 width:15px;\n"
"     margin: -7px -7px -7px -7px;                  \n"
"} \n"
"QSlider::add-page:horizontal{\n"
"background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
"\n"
"}\n"
"QSlider::sub-page:horizontal{                               \n"
" background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
"}")
        self.conf_Slider.setMaximum(100)
        self.conf_Slider.setValue(25)
        self.conf_Slider.setOrientation(Qt.Horizontal)
        self.conf_Slider.setTickPosition(QSlider.NoTicks)

        self.horizontalLayout_8.addWidget(self.conf_Slider)

        self.label_5 = QLabel(self.groupBox_9)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(90, 160, 61, 16))
        self.groupBox_10 = QGroupBox(self.tab_2)
        self.groupBox_10.setObjectName(u"groupBox_10")
        self.groupBox_10.setGeometry(QRect(30, 220, 961, 161))
        self.checkBox_3 = QCheckBox(self.groupBox_10)
        self.checkBox_3.setObjectName(u"checkBox_3")
        self.checkBox_3.setGeometry(QRect(76, 80, 91, 19))
        self.checkBox_4 = QCheckBox(self.groupBox_10)
        self.checkBox_4.setObjectName(u"checkBox_4")
        self.checkBox_4.setGeometry(QRect(226, 80, 111, 19))
        self.checkBox_5 = QCheckBox(self.groupBox_10)
        self.checkBox_5.setObjectName(u"checkBox_5")
        self.checkBox_5.setGeometry(QRect(405, 80, 131, 19))
        self.checkBox_6 = QCheckBox(self.groupBox_10)
        self.checkBox_6.setObjectName(u"checkBox_6")
        self.checkBox_6.setGeometry(QRect(600, 80, 121, 19))
        self.checkBox_8 = QCheckBox(self.groupBox_10)
        self.checkBox_8.setObjectName(u"checkBox_8")
        self.checkBox_8.setGeometry(QRect(760, 81, 121, 19))
        self.groupBox_11 = QGroupBox(self.tab_2)
        self.groupBox_11.setObjectName(u"groupBox_11")
        self.groupBox_11.setGeometry(QRect(30, 30, 961, 191))
        self.checkBox = QCheckBox(self.groupBox_11)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(862, 50, 91, 19))
        self.checkBox_2 = QCheckBox(self.groupBox_11)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(862, 114, 91, 19))
        self.groupBox_12 = QGroupBox(self.tab_2)
        self.groupBox_12.setObjectName(u"groupBox_12")
        self.groupBox_12.setGeometry(QRect(30, 620, 961, 171))
        self.rgbButton = QPushButton(self.groupBox_12)
        self.rgbButton.setObjectName(u"rgbButton")
        self.rgbButton.setGeometry(QRect(420, 90, 93, 28))
        self.tabWidget.addTab(self.tab_2, "")
        self.groupBox_11.raise_()
        self.groupBox_10.raise_()
        self.groupBox_7.raise_()
        self.groupBox_8.raise_()
        self.imgpath.raise_()
        self.videopath.raise_()
        self.imgsave.raise_()
        self.videosave.raise_()
        self.groupBox_9.raise_()
        self.groupBox_12.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1039, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.out_video.setText(QCoreApplication.translate("MainWindow", u"                                                       \u8bf7\u5148\u70b9\u51fb\u521d\u59cb\u5316\u6309\u94ae", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"\u6309\u952e\u63a7\u5236", None))
        self.pause.setText(QCoreApplication.translate("MainWindow", u"\u6682\u505c", None))
        self.Init.setText(QCoreApplication.translate("MainWindow", u"\u521d\u59cb\u5316", None))
        self.select_img.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u56fe\u7247", None))
        self.select_video.setText(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u89c6\u9891", None))
        self.exit.setText(QCoreApplication.translate("MainWindow", u"\u9000\u51fa", None))
        self.label_6.setText("")
        self.fps_label.setText("")
        self.msg.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b", None))
        self.groupBox_7.setTitle("")
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u56fe\u7247\u4fdd\u5b58\u8def\u5f84", None))
        self.groupBox_8.setTitle("")
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u89c6\u9891\u4fdd\u5b58\u8def\u5f84", None))
        self.imgpath.setText("")
        self.videopath.setText("")
        self.imgsave.setText(QCoreApplication.translate("MainWindow", u"\u6d4f\u89c8", None))
        self.videosave.setText(QCoreApplication.translate("MainWindow", u"\u6d4f\u89c8", None))
        self.groupBox_9.setTitle(QCoreApplication.translate("MainWindow", u"\u53c2\u6570\u8bbe\u7f6e", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u975e\u6781\u5927\u503c\u6291\u5236\u503c", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u7f6e\u4fe1\u5ea6", None))
        self.groupBox_10.setTitle(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u8bbe\u7f6e", None))
        self.checkBox_3.setText(QCoreApplication.translate("MainWindow", u"\u76ee\u6807\u68c0\u6d4b", None))
        self.checkBox_4.setText(QCoreApplication.translate("MainWindow", u"\u8f66\u9053\u7ebf\u68c0\u6d4b", None))
        self.checkBox_5.setText(QCoreApplication.translate("MainWindow", u"\u53ef\u884c\u9a76\u533a\u57df\u5212\u5206", None))
        self.checkBox_6.setText(QCoreApplication.translate("MainWindow", u"\u524d\u8f66\u8ddd\u79bb\u4f30\u8ba1", None))
        self.checkBox_8.setText(QCoreApplication.translate("MainWindow", u"\u6df1\u5ea6\u56fe", None))
        self.groupBox_11.setTitle(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u8bbe\u7f6e", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u56fe\u7247", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58\u89c6\u9891", None))
        self.groupBox_12.setTitle(QCoreApplication.translate("MainWindow", u"\u6807\u7b7eRGB\u8bbe\u7f6e", None))
        self.rgbButton.setText(QCoreApplication.translate("MainWindow", u"\u66f4\u6362\u989c\u8272", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e", None))
    # retranslateUi


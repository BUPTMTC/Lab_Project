# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1162, 712)
        self.gridLayout = QGridLayout(Form)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.ratioSlider = QSlider(Form)
        self.ratioSlider.setObjectName(u"ratioSlider")
        self.ratioSlider.setMaximumSize(QSize(16777215, 30))
        self.ratioSlider.setMaximum(100)
        self.ratioSlider.setOrientation(Qt.Horizontal)

        self.horizontalLayout.addWidget(self.ratioSlider)

        self.rlabel = QLabel(Form)
        self.rlabel.setObjectName(u"rlabel")
        self.rlabel.setMaximumSize(QSize(16777215, 30))
        font = QFont()
        font.setPointSize(15)
        self.rlabel.setFont(font)

        self.horizontalLayout.addWidget(self.rlabel)


        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.label = QLabel(Form)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(0, 650))

        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.rlabel.setText(QCoreApplication.translate("Form", u"Ratio:0.0001", None))
        self.label.setText("")
    # retranslateUi


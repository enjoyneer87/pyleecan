# -*- coding: utf-8 -*-

# File generated according to PBoreFlower.ui
# WARNING! All changes made in this file will be lost!
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ......GUI.Tools.FloatEdit import FloatEdit
from ......GUI.Dialog.DMachineSetup.DBore.WBoreOut.WBoreOut import WBoreOut

from pyleecan.GUI.Resources import pyleecan_rc


class Ui_PBoreFlower(object):
    def setupUi(self, PBoreFlower):
        if not PBoreFlower.objectName():
            PBoreFlower.setObjectName(u"PBoreFlower")
        PBoreFlower.resize(800, 470)
        PBoreFlower.setMinimumSize(QSize(800, 470))
        PBoreFlower.setMaximumSize(QSize(16777215, 16777215))
        self.horizontalLayout = QHBoxLayout(PBoreFlower)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.img_bore = QLabel(PBoreFlower)
        self.img_bore.setObjectName(u"img_bore")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_bore.sizePolicy().hasHeightForWidth())
        self.img_bore.setSizePolicy(sizePolicy)
        self.img_bore.setMinimumSize(QSize(400, 0))
        self.img_bore.setMaximumSize(QSize(16777215, 16777215))
        self.img_bore.setPixmap(
            QPixmap(u":/images/images/MachineSetup/LamParam/BoreFlower.png")
        )
        self.img_bore.setScaledContents(False)
        self.img_bore.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.img_bore)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.scrollArea = QScrollArea(PBoreFlower)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setMinimumSize(QSize(270, 200))
        self.scrollArea.setMaximumSize(QSize(270, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 268, 446))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.in_N = QLabel(self.scrollAreaWidgetContents)
        self.in_N.setObjectName(u"in_N")

        self.gridLayout.addWidget(self.in_N, 0, 0, 1, 1)

        self.si_N = QSpinBox(self.scrollAreaWidgetContents)
        self.si_N.setObjectName(u"si_N")

        self.gridLayout.addWidget(self.si_N, 0, 1, 1, 1)

        self.in_Rarc = QLabel(self.scrollAreaWidgetContents)
        self.in_Rarc.setObjectName(u"in_Rarc")

        self.gridLayout.addWidget(self.in_Rarc, 1, 0, 1, 1)

        self.lf_Rarc = FloatEdit(self.scrollAreaWidgetContents)
        self.lf_Rarc.setObjectName(u"lf_Rarc")

        self.gridLayout.addWidget(self.lf_Rarc, 1, 1, 1, 1)

        self.unit_Rarc = QLabel(self.scrollAreaWidgetContents)
        self.unit_Rarc.setObjectName(u"unit_Rarc")

        self.gridLayout.addWidget(self.unit_Rarc, 1, 2, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout)

        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.verticalLayout.addItem(self.verticalSpacer)

        self.w_out = WBoreOut(self.scrollAreaWidgetContents)
        self.w_out.setObjectName(u"w_out")

        self.verticalLayout.addWidget(self.w_out)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout.addWidget(self.scrollArea)

        self.retranslateUi(PBoreFlower)

        QMetaObject.connectSlotsByName(PBoreFlower)

    # setupUi

    def retranslateUi(self, PBoreFlower):
        PBoreFlower.setWindowTitle(
            QCoreApplication.translate("PBoreFlower", u"Form", None)
        )
        self.img_bore.setText("")
        self.in_N.setText(QCoreApplication.translate("PBoreFlower", u"N", None))
        self.in_Rarc.setText(QCoreApplication.translate("PBoreFlower", u"Rarc", None))
        self.unit_Rarc.setText(QCoreApplication.translate("PBoreFlower", u"m", None))

    # retranslateUi

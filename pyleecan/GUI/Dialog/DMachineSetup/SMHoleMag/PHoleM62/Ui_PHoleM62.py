# -*- coding: utf-8 -*-

# File generated according to PHoleM62.ui
# WARNING! All changes made in this file will be lost!
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ......GUI.Dialog.DMatLib.WMatSelect.WMatSelect import WMatSelect
from ......GUI.Tools.FloatEdit import FloatEdit

from pyleecan.GUI.Resources import pyleecan_rc


class Ui_PHoleM62(object):
    def setupUi(self, PHoleM62):
        if not PHoleM62.objectName():
            PHoleM62.setObjectName(u"PHoleM62")
        PHoleM62.resize(1164, 578)
        PHoleM62.setMinimumSize(QSize(0, 0))
        PHoleM62.setMaximumSize(QSize(16777215, 16777215))
        self.horizontalLayout = QHBoxLayout(PHoleM62)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.img_slot = QLabel(PHoleM62)
        self.img_slot.setObjectName(u"img_slot")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_slot.sizePolicy().hasHeightForWidth())
        self.img_slot.setSizePolicy(sizePolicy)
        self.img_slot.setMinimumSize(QSize(0, 0))
        self.img_slot.setMaximumSize(QSize(16777215, 16777215))
        self.img_slot.setPixmap(
            QPixmap(u":/images/images/MachineSetup/SMHoleMag/HoleM62_mag_int_rotor.png")
        )
        self.img_slot.setScaledContents(False)
        self.img_slot.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.img_slot)

        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.scrollArea = QScrollArea(PHoleM62)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setMinimumSize(QSize(270, 0))
        self.scrollArea.setMaximumSize(QSize(270, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 268, 554))
        self.verticalLayout_3 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.ck_is_radial = QCheckBox(self.scrollAreaWidgetContents)
        self.ck_is_radial.setObjectName(u"ck_is_radial")

        self.verticalLayout_3.addWidget(self.ck_is_radial)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.lf_W0 = FloatEdit(self.scrollAreaWidgetContents)
        self.lf_W0.setObjectName(u"lf_W0")

        self.gridLayout.addWidget(self.lf_W0, 2, 1, 1, 1)

        self.in_H0 = QLabel(self.scrollAreaWidgetContents)
        self.in_H0.setObjectName(u"in_H0")

        self.gridLayout.addWidget(self.in_H0, 0, 0, 1, 1)

        self.unit_H1 = QLabel(self.scrollAreaWidgetContents)
        self.unit_H1.setObjectName(u"unit_H1")

        self.gridLayout.addWidget(self.unit_H1, 1, 3, 1, 1)

        self.lf_H0 = FloatEdit(self.scrollAreaWidgetContents)
        self.lf_H0.setObjectName(u"lf_H0")

        self.gridLayout.addWidget(self.lf_H0, 0, 1, 1, 1)

        self.in_H1 = QLabel(self.scrollAreaWidgetContents)
        self.in_H1.setObjectName(u"in_H1")

        self.gridLayout.addWidget(self.in_H1, 1, 0, 1, 1)

        self.in_W0 = QLabel(self.scrollAreaWidgetContents)
        self.in_W0.setObjectName(u"in_W0")

        self.gridLayout.addWidget(self.in_W0, 2, 0, 1, 1)

        self.unit_H0 = QLabel(self.scrollAreaWidgetContents)
        self.unit_H0.setObjectName(u"unit_H0")

        self.gridLayout.addWidget(self.unit_H0, 0, 3, 1, 1)

        self.lf_H1 = FloatEdit(self.scrollAreaWidgetContents)
        self.lf_H1.setObjectName(u"lf_H1")

        self.gridLayout.addWidget(self.lf_H1, 1, 1, 1, 1)

        self.c_W0_unit = QComboBox(self.scrollAreaWidgetContents)
        self.c_W0_unit.addItem("")
        self.c_W0_unit.setObjectName(u"c_W0_unit")

        self.gridLayout.addWidget(self.c_W0_unit, 2, 3, 1, 1)

        self.verticalLayout_3.addLayout(self.gridLayout)

        self.w_mat_0 = WMatSelect(self.scrollAreaWidgetContents)
        self.w_mat_0.setObjectName(u"w_mat_0")
        self.w_mat_0.setMinimumSize(QSize(100, 0))

        self.verticalLayout_3.addWidget(self.w_mat_0)

        self.verticalSpacer_2 = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )

        self.verticalLayout_3.addItem(self.verticalSpacer_2)

        self.g_output = QGroupBox(self.scrollAreaWidgetContents)
        self.g_output.setObjectName(u"g_output")
        self.g_output.setMinimumSize(QSize(200, 0))
        self.verticalLayout = QVBoxLayout(self.g_output)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.out_slot_surface = QLabel(self.g_output)
        self.out_slot_surface.setObjectName(u"out_slot_surface")

        self.verticalLayout.addWidget(self.out_slot_surface)

        self.out_magnet_surface = QLabel(self.g_output)
        self.out_magnet_surface.setObjectName(u"out_magnet_surface")

        self.verticalLayout.addWidget(self.out_magnet_surface)

        self.verticalLayout_3.addWidget(self.g_output)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout.addWidget(self.scrollArea)

        QWidget.setTabOrder(self.lf_H0, self.lf_H1)
        QWidget.setTabOrder(self.lf_H1, self.lf_W0)

        self.retranslateUi(PHoleM62)

        QMetaObject.connectSlotsByName(PHoleM62)

    # setupUi

    def retranslateUi(self, PHoleM62):
        PHoleM62.setWindowTitle(QCoreApplication.translate("PHoleM62", u"Form", None))
        self.img_slot.setText("")
        self.ck_is_radial.setText(
            QCoreApplication.translate("PHoleM62", u"Radial", None)
        )
        self.in_H0.setText(QCoreApplication.translate("PHoleM62", u"H0", None))
        self.unit_H1.setText(QCoreApplication.translate("PHoleM62", u"m", None))
        self.in_H1.setText(QCoreApplication.translate("PHoleM62", u"H1", None))
        self.in_W0.setText(QCoreApplication.translate("PHoleM62", u"W0", None))
        self.unit_H0.setText(QCoreApplication.translate("PHoleM62", u"m", None))
        self.c_W0_unit.setItemText(
            0, QCoreApplication.translate("PHoleM62", u"m", None)
        )

        self.g_output.setTitle(QCoreApplication.translate("PHoleM62", u"Output", None))
        self.out_slot_surface.setText(
            QCoreApplication.translate("PHoleM62", u"Slot suface (2 part) : ?", None)
        )
        self.out_magnet_surface.setText(
            QCoreApplication.translate("PHoleM62", u"Single Magnet surface : ?", None)
        )

    # retranslateUi

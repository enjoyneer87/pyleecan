# -*- coding: utf-8 -*-

from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QFileDialog, QMessageBox, QWidget
from logging import getLogger
from multiprocessing import cpu_count

from .....loggers import GUI_LOG_NAME
from .....GUI import gui_option
from .....GUI.Dialog.DMachineSetup.SSimu.Gen_SSimu import Gen_SSimu
from .....Classes.MachineWRSM import MachineWRSM
from .....Classes.Simu1 import Simu1
from .....Classes.InputCurrent import InputCurrent
from .....Classes.OPdq import OPdq
from .....Classes.OPdqf import OPdqf
from .....Classes.MagFEMM import MagFEMM


class SSimu(Gen_SSimu, QWidget):
    """Step to define and run a simulation"""

    # Signal to DMachineSetup to know that the save popup is needed
    saveNeeded = Signal()  # No used here
    # Information for DMachineSetup nav
    step_name = "FEMM Simulation"

    def __init__(self, machine, material_dict, is_stator):
        """Initialize the GUI according to machine

        Parameters
        ----------
        self : SSimu
            A SSimu widget
        machine : Machine
            current machine to edit
        material_dict: dict
            Materials dictionary (library + machine)
        is_stator : bool
            To adapt the GUI to set either the stator or the rotor (unused)
        """

        # Build the interface according to the .ui file
        QWidget.__init__(self)
        self.setupUi(self)

        # Saving arguments
        self.machine = machine
        self.material_dict = material_dict

        # Adapt current widget
        self.in_I3.setHidden(not isinstance(self.machine, MachineWRSM))
        self.unit_I3.setHidden(not isinstance(self.machine, MachineWRSM))
        if self.machine.is_synchronous():
            self.in_I1.setText("Id")
            self.in_I2.setText("Iq")
            self.unit_I2.setText("[Arms]")
        else:
            self.in_I1.setText("I0")
            self.in_I2.setText("Phi0")
            self.unit_I2.setText("[rad]")
        self.unit_I1.setText("[Arms]")

        # Init default simulation to edit
        self.simu = Simu1(name="FEMM_" + self.machine.name)
        p = self.machine.get_pole_pair_number()
        Zs = self.machine.stator.slot.Zs
        self.simu.input = InputCurrent(Na_tot=2 * 3 * 5 * 7 * p, Nt_tot=10 * Zs * p)
        if isinstance(self.machine, MachineWRSM):
            self.simu.input.OP = OPdqf(N0=1000, Id_ref=0, Iq_ref=0, If_ref=5)
        else:
            self.simu.input.OP = OPdq(N0=1000, Id_ref=0, Iq_ref=0)
        self.simu.mag = MagFEMM(
            Kmesh_fineness=1,
            is_periodicity_a=True,
            is_periodicity_t=True,
            T_mag=20,
            nb_worker=cpu_count(),
        )
        self.simu.force = None

        # Init widget according to defaut simulation
        self.lf_N0.setValue(self.simu.input.OP.N0)
        self.lf_I1.setValue(0)
        self.lf_I2.setValue(0)
        self.lf_I3.setValue(5)  # Hidden if not used
        self.lf_Tmag.setValue(self.simu.mag.T_mag)
        self.si_Na_tot.setValue(self.simu.input.Na_tot)
        self.si_Nt_tot.setValue(self.simu.input.Nt_tot)
        self.is_per_a.setChecked(True)
        self.is_per_t.setChecked(True)
        self.lf_Kmesh.setValue(1)
        self.si_nb_worker.setValue(self.simu.mag.nb_worker)

        # Setup path result selection

        # Connecting the signal
        self.lf_N0.editingFinished.connect(self.set_N0)
        self.lf_I1.editingFinished.connect(self.set_Id_Iq)
        self.lf_I2.editingFinished.connect(self.set_Id_Iq)
        self.lf_I3.editingFinished.connect(self.set_I3)
        self.lf_Tmag.editingFinished.connect(self.set_Tmag)
        self.si_Na_tot.editingFinished.connect(self.set_Na_tot)
        self.si_Nt_tot.editingFinished.connect(self.set_Nt_tot)
        self.is_per_a.toggled.connect(self.set_per_a)
        self.is_per_t.toggled.connect(self.set_per_t)
        self.lf_Kmesh.editingFinished.connect(self.set_Kmesh)
        self.si_nb_worker.editingFinished.connect(self.set_nb_worker)

    def set_N0(self):
        """Update N0 according to the widget"""
        self.simu.input.OP.N0 = self.lf_N0.value()

    def set_Id_Iq(self):
        """Update Id/Iq according to the widget"""
        if self.machine.is_synchronous():
            self.simu.input.OP.Id_ref = self.lf_I1.value()
            self.simu.input.OP.Iq_ref = self.lf_I2.value()
        else:
            self.simu.input.OP.set_Id_Iq(I0=self.lf_I1.value(), Phi0=self.lf_I2.value())

    def set_I3(self):
        """Update If according to the widget"""
        self.simu.input.OP.If = self.lf_I3.value()

    def set_Tmag(self):
        """Update Tmag according to the widget"""
        self.simu.mag.T_mag = self.lf_Tmag.value()

    def set_Na_tot(self):
        """Update Na_tot according to the widget"""
        self.simu.input.Na_tot = self.si_Na_tot.value()

    def set_Nt_tot(self):
        """Update Nt_tot according to the widget"""
        self.simu.input.Nt_tot = self.si_Nt_tot.value()

    def set_per_a(self):
        """Update is_per_a according to the widget"""
        self.simu.mag.is_periodicity_a = self.is_per_a.isChecked()

    def set_per_t(self):
        """Update is_per_t according to the widget"""
        self.simu.mag.is_periodicity_t = self.is_per_t.isChecked()

    def set_Kmesh(self):
        """Update Kmesh according to the widget"""
        self.simu.mag.Kmesh_fineness = self.lf_Kmesh.value()

    def set_nb_worker(self):
        """Update nb_worker according to the widget"""
        self.simu.mag.nb_worker = self.si_nb_worker.value()

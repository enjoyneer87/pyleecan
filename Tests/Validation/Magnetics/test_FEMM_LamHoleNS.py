from os.path import join
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import pytest
from Tests import save_validation_path as save_path

from numpy import exp, sqrt, pi, max as np_max
from numpy.testing import assert_array_almost_equal

from pyleecan.Classes.OPdq import OPdq
from pyleecan.Classes.Simu1 import Simu1
from pyleecan.Classes.InputCurrent import InputCurrent
from pyleecan.Classes.LamHoleNS import LamHoleNS
from pyleecan.Classes.NotchEvenDist import NotchEvenDist
from pyleecan.Classes.MagFEMM import MagFEMM
from pyleecan.Classes.ForceMT import ForceMT

from pyleecan.Functions.load import load
from pyleecan.Functions.Plot import dict_2D

from pyleecan.definitions import DATA_DIR


@pytest.mark.long_5s
@pytest.mark.long_1m
@pytest.mark.MagFEMM
@pytest.mark.IPMSM
@pytest.mark.periodicity
@pytest.mark.SingleOP
def test_FEMM_LamHoleNS():
    """Validation of LamHoleNS in FEMM"""

    TP = load(join(DATA_DIR, "Machine", "Toyota_Prius.json"))
    TPNS = TP.copy()
    # Change rotor type to have different North/South Pole
    TPNS.rotor = LamHoleNS(init_dict=TP.rotor.as_dict())
    # North pole is unchange
    TPNS.rotor.hole_north = TP.rotor.hole
    # Change magnet dimensions on south pole
    TPNS.rotor.hole_south = [TP.rotor.hole[0].copy()]
    TPNS.rotor.hole_south[0].H3 *= 2
    TPNS.rotor.hole_south[0].H2 *= 8
    # First magnet of south pole have different material
    TPNS.rotor.hole_south[0].magnet_0.mat_type.name += "_2"
    TPNS.rotor.hole_south[0].magnet_0.mat_type.mag.mur_lin = 2

    # Check plot machine
    TPNS.plot(sym=4, save_path=join(save_path, "machine_sym.png"), is_show_fig=False)
    TPNS.plot(save_path=join(save_path, "machine_full.png"), is_show_fig=False)
    TPNS.rotor.plot(
        is_add_arrow=True,  # save_path=join(save_path, "rotor.png"), is_show_fig=False
    )
    plt.show()

    # Check periodicity
    assert TPNS.comp_periodicity_spatial() == (4, False)

    # Check machine in FEMM with sym
    simu = Simu1(name="test_FEMM_LamHoleNS", machine=TPNS)
    simu.path_result = join(save_path, simu.name)
    simu.input = InputCurrent(
        OP=OPdq(N0=1000, Id_ref=0, Iq_ref=0), Na_tot=2048, Nt_tot=1,
    )

    # Definition of the magnetic simulation: with periodicity
    simu.mag = MagFEMM(
        type_BH_stator=1,
        type_BH_rotor=1,
        is_periodicity_a=True,
        is_periodicity_t=False,
        nb_worker=cpu_count(),
        # Kmesh_fineness=2,
    )

    # Run simulations
    out = simu.run()

    out.mag.B.plot_2D_Data(
        "angle{°}",
        "time[0]",
        legend_list=["Periodic", "Full"],
        save_path=join(save_path, simu.name + "_B_space.png"),
        is_show_fig=False,
        **dict_2D
    )

    return out


# To run it without pytest
if __name__ == "__main__":

    out = test_FEMM_LamHoleNS()
    print("Done")

# %% [markdown]
# # Version information

# %%
from datetime import date
print("Running date:", date.today().strftime("%B %d, %Y"))
import pyleecan
print("Pyleecan version:" + pyleecan.__version__)
import SciDataTool
print("SciDataTool version:" + SciDataTool.__version__)


# %% [markdown]
# # Force Module을 사용하여 자기력을 계산하는 방법
# 
# 이 튜토리얼은 pyleecan으로 **자기력을 계산**하는 다양한 단계를 보여줍니다.
# 
# SciDataTool 개체의 기능과 사용법을 설명하기 위해 FEMM뮬레이션을 실행하고 전류를 인가하여 실행 시간을 줄이기 위해 주기성과 병렬연산을 사용합니다.

# %%

# Load the machine
from os.path import join
from numpy import (
    zeros,
    exp,
    pi,
    real,
    meshgrid,
    mean,
)

# 힘 계산 클래스 추가
from pyleecan.Classes.ForceMT import ForceMT
from pyleecan.Classes.OPdq import OPdq
from pyleecan.Classes.Simu1 import Simu1
from pyleecan.Classes.MagFEMM import MagFEMM
from pyleecan.Classes.InputCurrent import InputCurrent
from pyleecan.definitions import DATA_DIR


from pyleecan.Functions.load import load
from pyleecan.Functions.Plot import dict_2D, dict_3D 

# print(DATA_DIR)
path=r'Z:\Thesis\HDEV\06_Pyleecan'

# path = 'D:/NGV/Lecture_Simu_Model'
modelname = 'HDEV_nskew'
model = join(modelname+'.json')

# Load machine
HDEV12p = load(join(path,model))
# HDEV12p.plot()
save_path=path


# %%
import numpy as np
from pyleecan.Classes.VarLoadCurrent import VarLoadCurrent
from pyleecan.Classes.ImportMatrixVal import ImportMatrixVal
from pyleecan.Classes.InputCurrent import InputCurrent
from pyleecan.Classes.ImportGenVectLin import ImportGenVectLin
from pyleecan.Classes.ImportGenVectSin import ImportGenVectSin
from pyleecan.Classes.ImportGenMatrixSin import ImportGenMatrixSin
ref_simu = Simu1(name="HDEV12pcurrent_sweep", machine=HDEV12p)



# %% [markdown]
# ## Simu 1 Simulation (No-load)

# %%
from numpy import sqrt, pi
# Definition of the magnetic simulation
# ref_simu.mag = MagFEMM(
#     type_BH_stator=0,
#     type_BH_rotor=0,
#     is_periodicity_a=True,
#     is_periodicity_t=True,
#     Kgeo_fineness=1,
#     Kmesh_fineness=2,
#     nb_worker=8,
# )
ref_simu.mag = MagFEMM(
    type_BH_stator=0,
    type_BH_rotor=0,
    is_periodicity_a=True,
    is_periodicity_t=True,
    Kgeo_fineness=0.2,
    Kmesh_fineness=0.2,
    nb_worker=8,
)
ref_simu.force=ForceMT(  is_periodicity_a=True,is_periodicity_t=True)




ref_simu.input = InputCurrent()
# ref_simu.input.Nt_tot = 720 # Number of time step
ref_simu.input.Nt_tot = 120 # Number of time step
ref_simu.input.Na_tot = 2520 # Spatial discretization
# Set reference simulation operating point
ref_simu.input.OP = OPdq(N0 = 1000) # Rotor speed [rpm]
ref_simu.input.OP.Tem_av_ref = 353
ref_simu.input.OP.set_I0_Phi0(I0=250/sqrt(2), Phi0=140*pi/180)




# %%
output_ref = ref_simu.run()
# Flux
# output_ref.mag.B.plot_2D_Data("angle","time[1]",component_list=["radial"])

# Torque
# output_ref.mag.Tem.plot_2D_Data("time")
print("Main torque Output:")
print("Average Electromagnetic torque [N.m]:" +str(output_ref.mag.Tem_av))
print("Peak to Peak Torque ripple [N.m]:" +str(output_ref.mag.Tem_rip_pp))
print("Peak to Peak Torque ripple normalized according to average torque [-]:" +str(output_ref.mag.Tem_rip_norm))
print("Torque values (with symmetry):\n"+str(output_ref.mag.Tem.values))

# Operating point
print("\nOperating Point:")
print("Rotor speed [rpm]:"+str(output_ref.elec.OP.N0))
print("Id [Arms]:"+str(output_ref.elec.OP.Id_ref))
print("Iq [Arms]:"+str(output_ref.elec.OP.Iq_ref))

# %% [markdown]
# ## OP_Matrix  작성 VarloadCurrent>VarLoad(OP_matrix)>VarSimu

# %%
from pyleecan.Classes.VarLoadCurrent import VarLoadCurrent
from numpy import zeros, ones, linspace, array, sqrt, arange

# Definition of the enforced output of the electrical module
# Is_mat = zeros((1, 3))
# Is_mat[0, :] = np.array([0, 12.2474, -12.2474])
# Is = ImportMatrixVal(value=Is_mat)
# time = ImportGenVectLin(start=0, stop=0, num=1, endpoint=False)
# Na_tot = 2048
# simu_vop = ref_simu.copy()
# simu_vop.mag.import_file = None
# varload = VarLoadCurrent()
# simu_vop.var_simu = varload
# N_speed=50


# Build OP_matrix with a meshgrid of Id/Iq
n_Id=6
n_Iq=6
# Id_min, Id_max = -750, 750
# Iq_min, Iq_max = -750, 750
Id_min, Id_max = -750, 0
Iq_min, Iq_max = 0, 750
Id, Iq = np.meshgrid(
np.linspace(Id_min, Id_max, n_Id),np.linspace(Iq_min, Iq_max, n_Iq)
)
# Id Iq type
OP_matrix = np.zeros((n_Id * n_Iq, 3))
OP_matrix[:, 0] = ref_simu.input.OP.N0
OP_matrix[:, 1] = Id.ravel()/np.sqrt(2) #ravel() 다차원 배열 1차원 변환 함수
OP_matrix[:, 2] = Iq.ravel()/np.sqrt(2)
# type(OP_matrix)

# Ipk phase type
# Creating the Operating point matrix
# OP_matrix = zeros((N_speed,4))
# # Set N0 = 2000 [rpm] for all simulation
# OP_matrix[:,0] = 2000 * ones((N_speed))
# # Set I0 = 250 / sqrt(2) [A] (RMS) for all simulation
# OP_matrix[:,1] = 250/sqrt(2) * ones((N_speed)) 
# Set Phi0 from 60° to 180°
# OP_matrix[:,2] = Phi0_ref
# Set reference torque from Yang et al, 2013
# OP_matrix[:,3] = Tem_av_ref

varload = VarLoadCurrent()
varload.OP_matrix = OP_matrix
varload.is_reuse_femm_file=True  

# %% [markdown]
# ### Simu1에 varload 할당

# %%
from pyleecan.Classes.VarSimu import VarSimu
simu_vop = ref_simu.copy()
simu_vop.var_simu = varload
simu_vop.var_simu.is_keep_all_output = True

# %%
# Xout_vop = simu_vop.run()


# # %% [markdown]
# # ### output] Xout_vop Export to Mat

# %%
import os
from datetime import datetime
import scipy.io as sio

output_folder = r'Z:\Thesis\HDEV\06_Pyleecan'
run_simulation_with_variable_Tmag(120,output_folder)
run_simulation_with_variable_Tmag(85,output_folder)
run_simulation_with_variable_Tmag(65,output_folder)
run_simulation_with_variable_Tmag(40,output_folder)
run_simulation_with_variable_Tmag(0,output_folder)
run_simulation_with_variable_Tmag(-20,output_folder)
run_simulation_with_variable_Tmag(-40,output_folder)


import os

def run_simulation_with_variable_Tmag(T_mag_value, output_folder):
    # Create T_mag folder
    if T_mag_value < 0:
        t_mag_folder = os.path.join(output_folder, 'T_magminus{}'.format(abs(T_mag_value)))
    else:
        t_mag_folder = os.path.join(output_folder, 'T_mag{}'.format(T_mag_value))
    os.makedirs(t_mag_folder, exist_ok=True)
    
    extension = '.mat'

    
    simu_vopTmag = simu_vop.copy()
    simu_vopTmag.mag.T_mag = T_mag_value
    Xout_vopTmag = simu_vopTmag.run()

    force_r_r0_0fs = []
    force_r_r0_6fs = []
    force_r_r0_12fs = []
    force_r_r24_6fs = []
    force_t_r0_0fs = []
    force_t_r0_6fs = []
    force_t_r0_12fs = []
    force_t_r24_6fs = []

    Torque_map = []

    for i in range(0, len(Xout_vopTmag.output_list)):
        Torque_map = np.append(Torque_map, Xout_vopTmag.output_list[i].mag.Tem_av)

        filename_prefix = datetime.now().strftime("%Y%m%d_%H")
        filename = '{}_Tmag{}op_case{}'.format(filename_prefix, T_mag_value, i)
        filepath = os.path.join(t_mag_folder, filename + extension)
        Xout_vopTmag.output_list[i].export_to_mat(filepath)

        force_r_r0_0fs = np.append(force_r_r0_0fs, Xout_vopTmag.output_list[i].force.AGSF.components['radial'].get_magnitude_along('freqs->elec_order=0', 'wavenumber=0')['AGSF_r'])
        force_r_r0_6fs = np.append(force_r_r0_6fs, Xout_vopTmag.output_list[i].force.AGSF.components['radial'].get_magnitude_along('freqs->elec_order=6', 'wavenumber=0')['AGSF_r'])
        force_r_r0_12fs = np.append(force_r_r0_12fs, Xout_vopTmag.output_list[i].force.AGSF.components['radial'].get_magnitude_along('freqs->elec_order=12', 'wavenumber=0')['AGSF_r'])
        force_r_r24_6fs = np.append(force_r_r24_6fs, Xout_vopTmag.output_list[i].force.AGSF.components['radial'].get_magnitude_along('freqs->elec_order=6', 'wavenumber=24')['AGSF_r'])
    
        force_t_r0_0fs = np.append(force_t_r0_0fs, Xout_vopTmag.output_list[i].force.AGSF.components['tangential'].get_magnitude_along('freqs->elec_order=0', 'wavenumber=0')['AGSF_t'])
        force_t_r0_6fs = np.append(force_t_r0_6fs, Xout_vopTmag.output_list[i].force.AGSF.components['tangential'].get_magnitude_along('freqs->elec_order=6', 'wavenumber=0')['AGSF_t'])
        force_t_r0_12fs = np.append(force_t_r0_12fs, Xout_vopTmag.output_list[i].force.AGSF.components['tangential'].get_magnitude_along('freqs->elec_order=12', 'wavenumber=0')['AGSF_t'])
        force_t_r24_6fs = np.append(force_t_r24_6fs, Xout_vopTmag.output_list[i].force.AGSF.components['tangential'].get_magnitude_along('freqs->elec_order=6', 'wavenumber=24')['AGSF_t'])

        Force2DFFTMap = {
            'force_r_r0_0fs': force_r_r0_0fs,
            'force_r_r0_6fs': force_r_r0_6fs,
            'force_r_r0_12fs': force_r_r0_12fs,
            'force_r_r24_6fs': force_r_r24_6fs,
            'force_t_r0_0fs': force_t_r0_0fs,
            'force_t_r0_6fs': force_t_r0_6fs,
            'force_t_r0_12fs': force_t_r0_12fs,
            'force_t_r24_6fs': force_t_r24_6fs
        }

        filename = '{}_Tmag{}Force{}'.format(filename_prefix, T_mag_value, i)
        filepath = os.path.join(t_mag_folder, filename + extension)
        sio.savemat(filepath, Force2DFFTMap)



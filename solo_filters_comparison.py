import os
import sys
import numpy as np
import pinocchio as pin
from example_robot_data import load
import matplotlib.pyplot as plt

from contact_forces_estimator import ContactForcesEstimator
from kalman_filters import ImuLegKF, cross3
from complementary_filter import Estimator

from data_readers import read_data_file_laas, read_data_file_laas_ctrl, shortened_arr_dic, add_measurement_noise

class Device:
    """
    Notation adapter
    """
    def __init__(self, i_a_oi, i_omg_oi, o_q_i, qa, dqa):
        self.baseLinearAcceleration = i_a_oi
        self.baseAngularVelocity = i_omg_oi
        self.baseOrientation = o_q_i
        self.q_mes = qa
        self.v_mes = dqa
        
##############################################
# extract raw trajectory from data file
##############################################
# dt = 1e-3  # real robot
dt = 2e-3  # simu pybullet

cwdir = os.getcwd()
DATA_FOLDER_RESULTS = os.path.join(cwdir, 'data/quadruped_experiments_results/')
if not os.path.exists(DATA_FOLDER_RESULTS):
    os.makedirs(DATA_FOLDER_RESULTS)
DATA_FOLDER = os.path.join(cwdir, 'data/')

# data_file_meas = 'Logs_05_10_2020_18h/data_2020_11_05_18_18.npz'
# data_file_ctrl = 'Logs_05_10_2020_18h/data_control_2020_11_05_18_18.npz'


# data_file_meas = 'Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_17.npz'  # stamping
# data_file_meas = 'Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_18.npz'  # sin
# data_file_meas = 'Experiments_Replay_30_11_2020_bis/data_2020_11_30_17_22.npz'  # walking
data_file_ctrl = 'Experiments_Replay_30_11_2020_bis/data_control_2020_11_30_17_22.npz'

# data_file_meas = 'Point_feet_27_11_20/data_2020_11_27_14_46.npz'
# data_file_ctrl = 'Logs_05_10_2020_18h/data_control_2020_11_05_18_18.npz'

# data_file_meas = 'SinStamping_Corrected_09_12_2020/data_2020_12_09_17_54.npz'  # sin
# data_file_meas = 'SinStamping_Corrected_09_12_2020/data_2020_12_09_17_56.npz'  # stamping

# data_file_meas = 'Experiments_Walk_17_12_2020/data_2020_12_17_14_25.npz'  
# data_file_ctrl = 'Experiments_Walk_17_12_2020/data_control_2020_12_17_14_25.npz'  
# data_file_meas = 'Experiments_Walk_17_12_2020/data_2020_12_17_14_29.npz'  
# data_file_ctrl = 'Experiments_Walk_17_12_2020/data_control_2020_12_17_14_29.npz' 

# Simulation
# data_file_meas = 'Simulation_Walk_06_01_2020/data_2021_01_06_17_47.npz'
# data_file_ctrl = 'Simulation_Walk_06_01_2020/data_control_2021_01_06_17_47.npz'
# data_file_meas = 'Simulation_Walk_06_01_2020/data_2021_01_06_17_48.npz'
# data_file_ctrl = 'Simulation_Walk_06_01_2020/data_control_2021_01_06_17_48.npz'
# data_file_meas = 'Simulation_Walk_08_01_2021/data_2021_01_08_13_46.npz'
# data_file_ctrl = 'Simulation_Walk_08_01_2021/data_control_2021_01_08_13_46.npz'



CTRL_FILE = True  # use the available control file for contacts
THRESH_FZ = 1  # sin


print('Reading ', DATA_FOLDER+data_file_meas)
arr_dic_meas = read_data_file_laas(DATA_FOLDER+data_file_meas, dt)
arr_dic_meas = add_measurement_noise(arr_dic_meas)
print('Reading ', DATA_FOLDER+data_file_ctrl)
arr_dic_ctrl = read_data_file_laas_ctrl(DATA_FOLDER+data_file_ctrl)

# Shorten?
arr_dic_meas = shortened_arr_dic(arr_dic_meas, N=5000)
arr_dic_ctrl = shortened_arr_dic(arr_dic_ctrl, N=5000)

t_arr = arr_dic_meas['t']
N = len(t_arr)

###########################
# initialize the estimators
###########################
# initial state
# position: 0,0,0
# velocity: 0,0,0 -> HYP robot does not move
# height of the robot base frame
# H_INIT = 0.235  # with urdf feet
# H_INIT = 0.22294615  # with smaller black feet
H_INIT = 0.205  # with point feet

o_p_ob = np.array([0,0,H_INIT])
o_q_i = arr_dic_meas['o_q_i'][0,:]
o_v_oi = np.zeros(3)
qa = arr_dic_meas['qa'][0,:]
q_init = np.hstack([o_p_ob, o_q_i, qa])

KFImuLegWithFeet = ImuLegKF(dt, q_init)
cf = Estimator(dt, N, h_init=H_INIT)
KFImuLeg = Estimator(dt, N, h_init=H_INIT, kf_enabled=True)

robot = load('solo12')
force_est = ContactForcesEstimator(robot, KFImuLegWithFeet.contact_ids, dt)

# some useful recordings
q_kf_arr = np.zeros((N, 19))
v_kf_arr = np.zeros((N, 18))
q_cf_arr = np.zeros((N, 19))
v_cf_arr = np.zeros((N, 18))
q_kfwof_arr = np.zeros((N, 19))
v_kfwof_arr = np.zeros((N, 18))
fz_arr = np.zeros((N,4))
contact_status_arr = np.zeros((N,4))
feet_state_arr = np.zeros((N,4*3))

for i in range(N):
    # define measurements
    o_R_i = arr_dic_meas['o_R_i'][i,:]  # retrieve IMU pose estimation
    o_q_i = arr_dic_meas['o_q_i'][i,:]  # retrieve IMU pose estimation
    i_a_oi = arr_dic_meas['i_a_oi'][i,:] # retrieve IMU gravity compensated linear acceleration expressed in IMU (robot) frame
    o_a_oi = arr_dic_meas['o_a_oi'][i,:] # retrieve IMU gravity compensated linear acceleration expressed in world frame

    i_omg_oi = arr_dic_meas['i_omg_oi'][i,:]
    qa = arr_dic_meas['qa'][i,:]
    dqa = arr_dic_meas['dqa'][i,:]

    if CTRL_FILE:
        contact_status = arr_dic_meas['contactStatus'][i,:]
        goals = arr_dic_ctrl['log_feet_pos_target'][:,:,i]
    else:
        # contact estimation based on force estimation
        ddqa = np.zeros(12)
        i_domg_oi = np.zeros(3)
        tauj = arr_dic_meas['tau'][i,:]
        o_forces = force_est.compute_contact_forces2(qa, dqa, ddqa, o_R_i, i_omg_oi, i_domg_oi, o_a_oi, tauj, world_frame=True)
        fz_arr[i,:] = o_forces[:,2]
        contact_status = np.array([fz > THRESH_FZ for fz in o_forces[:,2]])  # simple contact detection
        contact_status_arr[i,:] = contact_status 

        # initial blip
        # if i*dt < 0.3:
        #     contact_status = np.zeros(4)

        goals = np.zeros((3,4))

    # Kalman Filter with feet
    KFImuLegWithFeet.run_filter(o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status)
    q_kf_arr[i,:], v_kf_arr[i,:] = KFImuLegWithFeet.get_configurations()
    feet_state_arr[i,:] = KFImuLegWithFeet.get_state()[6:]

    # Complementary filter
    device = Device(i_a_oi, i_omg_oi, o_q_i, qa, dqa)
    cf.run_filter(i, contact_status, device, goals, remaining_steps=100)
    q_cf_arr[i,:], v_cf_arr[i,:] = cf.get_configurations()

    # Kalman Filter without feet
    device = Device(i_a_oi, i_omg_oi, o_q_i, qa, dqa)
    KFImuLeg.run_filter(i, contact_status, device, goals, remaining_steps=100)
    q_kfwof_arr[i,:], v_kfwof_arr[i,:] = KFImuLeg.get_configurations()


# data to copy
res_arr_dic = {}
copy_lst = ['t', 'w_v_wm', 'm_v_wm', 'w_q_m', 'w_p_wm']
for k in copy_lst:
    res_arr_dic[k] = arr_dic_meas[k]
# add estimated data
res_arr_dic['q_kf'] = q_kf_arr
res_arr_dic['v_kf'] = v_kf_arr
res_arr_dic['q_cf'] = q_cf_arr
res_arr_dic['v_cf'] = v_cf_arr
res_arr_dic['q_kfwof'] = q_kfwof_arr
res_arr_dic['v_kfwof'] = v_kfwof_arr

out_path = DATA_FOLDER_RESULTS+'out.npz'
np.savez(out_path, **res_arr_dic)
print(out_path, ' saved')


# some plots
plt.figure('Normal forces')
plt.title('Normal forces')
for i in range(4):
    plt.subplot(4,1,i+1)
    plt.plot(t_arr, fz_arr[:,i], label='fz '+KFImuLegWithFeet.contact_frame_names[i])
    plt.plot(t_arr, contact_status_arr[:,i]*THRESH_FZ, label='contact')
    plt.plot(t_arr, arr_dic_meas['contactStatus'][:,i]*THRESH_FZ, label='contact')
    plt.hlines(0, t_arr[0]-1, t_arr[-1]+1, 'k')
plt.legend()

plt.figure('Filter positions')
plt.title('Filter positions')
for i in range(3):
    plt.subplot(3,1,1+i)
    plt.plot(t_arr, q_kf_arr[:,i], 'g', label='KFWithFeet')
    plt.plot(t_arr, q_cf_arr[:,i], 'b', label='CF')
    plt.plot(t_arr, q_kfwof_arr[:,i], 'r', label='KFWithoutFeet')
    plt.legend()

plt.figure('KF feet XYZ')
plt.title('KF feet XYZ')
for i in range(3):
    plt.subplot(3,1,1+i)
    for j in range(4):
        plt.plot(t_arr, feet_state_arr[:,3*j+i], label=str(j))
    plt.legend()

plt.figure('KF feet XY')
plt.title('KF feet XY')
for i_ee in range(4):
    plt.plot(feet_state_arr[:,3*i_ee], feet_state_arr[:,3*i_ee+1], '.', markersize=1, label=KFImuLegWithFeet.contact_frame_names[i_ee])
    plt.legend()
plt.plot(q_kf_arr[:,0], q_kf_arr[:,1], label='base')
plt.legend()

if len(KFImuLegWithFeet.o_v_oi_dic[0]) > 0:
    plt.figure('Base vel from feet vs mocap')
    plt.title('Base vel from feet vs mocap')
    for i_ee in range(4):
        plt.plot(t_arr, KFImuLegWithFeet.o_v_oi_dic[i_ee], '.', markersize=1, label=KFImuLegWithFeet.contact_frame_names[i_ee])
        plt.legend()
    plt.plot(q_kf_arr[:,0], q_kf_arr[:,1], label='base')
    plt.legend()

    for i_ee in range(4):
        KFImuLegWithFeet.o_v_oi_dic[i_ee] = np.array(KFImuLegWithFeet.o_v_oi_dic[i_ee])
    plt.figure('Base velX from feet vs mocap')
    plt.title('Base velX from feet vs mocap')
    l = [0]
    for i_ee in l:
        plt.plot(t_arr, KFImuLegWithFeet.o_v_oi_dic[i_ee][:,0], '.', markersize=1, label=KFImuLegWithFeet.contact_frame_names[i_ee])
    # plt.plot(t_arr, arr_dic_meas['m_v_wm'][:,0], 'rx', label='Mo-Cap')
    # plt.legend()

if '--show' in sys.argv:
    plt.show()
    


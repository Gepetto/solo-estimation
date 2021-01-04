import os
import sys
import numpy as np
import pinocchio as pin
from example_robot_data import load
import matplotlib.pyplot as plt

from contact_forces_estimator import ContactForcesEstimator
from kalman_filters import ImuLegKF, cross3
from complementary_filter import Estimator

from data_readers import read_data_file_laas, read_data_file_laas_ctrl, shortened_arr_dic

class Device:
    """
    Notation adapter
    """
    def __init__(self, o_a_oi, i_omg_oi, o_q_i, qa, dqa):
        self.baseLinearAcceleration = o_a_oi
        self.baseAngularVelocity = i_omg_oi
        self.baseOrientation = o_q_i
        self.q_mes = qa
        self.v_mes = dqa
        
##############################################
# extract raw trajectory from data file
##############################################
dt = 1e-3

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
data_file_meas = 'SinStamping_Corrected_09_12_2020/data_2020_12_09_17_56.npz'  # stamping

CTRL_FILE = False  # use the available control file for contacts
THRESH_FZ = 1  # sin


print('Reading ', DATA_FOLDER+data_file_meas)
arr_dic_meas = read_data_file_laas(DATA_FOLDER+data_file_meas, dt)
print('Reading ', DATA_FOLDER+data_file_ctrl)
arr_dic_ctrl = read_data_file_laas_ctrl(DATA_FOLDER+data_file_ctrl)

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

kf = ImuLegKF(dt, q_init)
cf = Estimator(dt, N, h_init=H_INIT)

robot = load('solo12')
force_est = ContactForcesEstimator(robot, kf.contact_ids, dt)

# some useful recordings
q_kf_arr = np.zeros((N, 19))
v_kf_arr = np.zeros((N, 18))
q_cf_arr = np.zeros((N, 19))
v_cf_arr = np.zeros((N, 18))
fz_arr = np.zeros((N,4))
contact_status_arr = np.zeros((N,4))
feet_state_arr = np.zeros((N,4*3))

for i in range(N):
    # define measurements
    o_R_i = arr_dic_meas['o_R_i'][i,:]  # retrieve IMU pose estimation
    o_q_i = arr_dic_meas['o_q_i'][i,:]  # retrieve IMU pose estimation
    o_a_oi = arr_dic_meas['o_a_oi'][i,:] # retrieve IMU linear acceleration estimation

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
        if i*dt < 0.3:
            contact_status = np.zeros(4)

        goals = np.zeros((3,4))

    # Kalman Filter
    kf.run_filter(o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status)
    q_kf_arr[i,:], v_kf_arr[i,:] = kf.get_configurations()
    feet_state_arr[i,:] = kf.get_state()[6:]

    # Complementary filter
    device = Device(o_a_oi, i_omg_oi, o_q_i, qa, dqa)
    cf.run_filter(i, contact_status, device, goals, remaining_steps=100)
    q_cf_arr[i,:], v_cf_arr[i,:] = cf.get_configurations()


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


out_path = DATA_FOLDER_RESULTS+'out.npz'
np.savez(out_path, **res_arr_dic)
print(out_path, ' saved')


# some plots
plt.figure('Normal forces')
plt.title('Normal forces')
for i in range(4):
    plt.subplot(4,1,i+1)
    plt.plot(t_arr, fz_arr[:,i], label='fz '+kf.contact_frame_names[i])
    plt.plot(t_arr, contact_status_arr[:,i]*THRESH_FZ, label='contact')
    plt.plot(t_arr, arr_dic_meas['contactStatus'][:,i]*THRESH_FZ, label='contact')
    plt.hlines(0, t_arr[0]-1, t_arr[-1]+1, 'k')
plt.legend()

plt.figure('Filter positions')
plt.title('Filter positions')
for i in range(3):
    plt.subplot(3,1,1+i)
    plt.plot(t_arr, q_kf_arr[:,i], 'g', label='KF')
    plt.plot(t_arr, q_cf_arr[:,i], 'b', label='CF')
    plt.legend()

plt.figure('KF feet')
plt.title('KF feet')
for i in range(3):
    plt.subplot(3,1,1+i)
    for j in range(4):
        plt.plot(t_arr, feet_state_arr[:,3*j+i], label=str(j))
    plt.legend()



if '--show' in sys.argv:
    plt.show()
    


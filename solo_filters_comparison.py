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
os.makedirs(DATA_FOLDER_RESULTS)
DATA_FOLDER = os.path.join(cwdir, 'data/')

data_file_meas = 'Logs_05_10_2020_18h/data_2020_11_05_18_18.npz'
data_file_ctrl = 'Logs_05_10_2020_18h/data_control_2020_11_05_18_18.npz'

print('Reading ', DATA_FOLDER+data_file_meas)
arr_dic_meas = read_data_file_laas(DATA_FOLDER+data_file_meas, dt)
print('Reading ', DATA_FOLDER+data_file_ctrl)
arr_dic_ctrl = read_data_file_laas_ctrl(DATA_FOLDER+data_file_ctrl)
# arr_dic_meas = shortened_arr_dic(arr_dic_meas, 0, 2000)

t_arr = arr_dic_meas['t']
N = len(t_arr)

###########################
# initialize the estimators
###########################
# initial state
# position: 0,0,0
# velocity: 0,0,0 -> HYP robot does not move
o_p_oi = np.zeros(3)
o_q_i = arr_dic_meas['o_q_i'][0,:]
o_v_oi = np.zeros(3)
qa = arr_dic_meas['qa'][0,:]
q_init = np.hstack([o_p_oi, o_q_i, qa])

kf = ImuLegKF(dt, q_init)
cf = Estimator(dt, N)

# some useful recordings
q_arr_kf = np.zeros((N, 19))
v_arr_kf = np.zeros((N, 18))
q_arr_cf = np.zeros((N, 19))
v_arr_cf = np.zeros((N, 18))

for i in range(N):
    # define measurements
    o_R_i = arr_dic_meas['o_R_i'][i,:]  # retrieve IMU pose estimation
    o_q_i = arr_dic_meas['o_q_i'][i,:]  # retrieve IMU pose estimation
    o_a_oi = arr_dic_meas['o_a_oi'][i,:] # retrieve IMU linear acceleration estimation

    i_omg_oi = arr_dic_meas['i_omg_oi'][i,:]
    qa = arr_dic_meas['qa'][i,:]
    dqa = arr_dic_meas['dqa'][i,:]

    contact_status = arr_dic_meas['contactStatus'][i,:]
    goals = arr_dic_ctrl['log_feet_pos_target'][:,:,i]

    # Kalman Filter
    kf.run_filter(o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status)
    q_arr_kf[i,:], v_arr_kf[i,:] = kf.get_configurations()

    # Complementary filter
    device = Device(o_a_oi, i_omg_oi, o_q_i, qa, dqa)
    cf.run_filter(i, contact_status, device, goals)
    q_arr_cf[i,:], v_arr_cf[i,:] = (conf for conf in cf.get_configurations())


# data to copy
res_arr_dic = {}
copy_lst = ['t', 'w_v_wm', 'm_v_wm', 'w_q_m', 'w_p_wm']
for k in copy_lst:
    res_arr_dic[k] = arr_dic_meas[k]
# add estimated data
res_arr_dic['q_kf'] = q_arr_kf
res_arr_dic['v_kf'] = v_arr_kf
res_arr_dic['q_cf'] = q_arr_cf
res_arr_dic['v_cf'] = v_arr_cf


out_path = DATA_FOLDER_RESULTS+'out.npz'
np.savez(out_path, **res_arr_dic)
print(out_path, ' saved')

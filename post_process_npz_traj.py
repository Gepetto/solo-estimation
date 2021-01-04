#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import datetime
from scipy import signal
import pinocchio as pin
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# uses https://github.com/uzh-rpg/rpg_trajectory_evaluation.git
# sys.path.append('/home/mfourmy/Documents/Phd_LAAS/installations/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation') 
# import trajectory as rpg_traj

from data_readers import shortened_arr_dic

cwdir = os.getcwd()
DATA_FOLDER_RESULTS = os.path.join(cwdir, 'data/quadruped_experiments_results/')
print(DATA_FOLDER_RESULTS)

# from wolf estimation
data_file = 'out.npz'
data_file_post = 'out_post.npz'

# Keys:
print('Reading ', DATA_FOLDER_RESULTS+data_file)
arr_dic = np.load(DATA_FOLDER_RESULTS+data_file)
N = len(arr_dic['t'])
arr_dic = shortened_arr_dic(arr_dic, 0, N-200)

dt = 1e-3
t_arr = arr_dic['t']
N = len(t_arr)
print('N: ', N)

# EST
q_kf_arr = arr_dic['q_kf']
v_kf_arr = arr_dic['v_kf']
q_cf_arr = arr_dic['q_cf']
v_cf_arr = arr_dic['v_cf']

o_p_ob_kf_arr = q_kf_arr[:,:3]
o_p_ob_cf_arr = q_cf_arr[:,:3]
o_q_b_kf_arr = q_kf_arr[:,3:7]
o_q_b_cf_arr = q_cf_arr[:,3:7]

b_v_ob_kf_arr = v_kf_arr[:,:3]
b_v_ob_cf_arr = v_cf_arr[:,:3]

# GT
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']
w_v_wm_arr = arr_dic['w_v_wm']
m_v_wm_arr = arr_dic['m_v_wm']

# plot before trajectory alignment
plt.figure('P KF CF before alignment')
plt.plot(w_p_wm_arr[:,0], w_p_wm_arr[:,1], label='MOCAP')
plt.plot(q_kf_arr[:,0], q_kf_arr[:,1], label='KF')
plt.plot(q_cf_arr[:,0], q_cf_arr[:,1], label='CF')
plt.legend()
plt.grid()
# plt.show()

# Trajectory alignment 
# based on first frame (se3 alignment)
o_p_ob_kf_init = o_p_ob_kf_arr[0,:]
o_p_ob_cf_init = o_p_ob_cf_arr[0,:]
w_p_wm_init = w_p_wm_arr[0,:]

w_R_m_init = pin.Quaternion(w_q_m_arr[0,:].reshape((4,1))).toRotationMatrix()
o_R_b_kf_init = pin.Quaternion(o_q_b_kf_arr[0,:].reshape((4,1))).toRotationMatrix()
o_R_b_cf_init = pin.Quaternion(o_q_b_cf_arr[0,:].reshape((4,1))).toRotationMatrix()

w_T_m_init = pin.SE3(w_R_m_init, w_p_wm_init)
o_T_b_kf_init = pin.SE3(o_R_b_kf_init, o_p_ob_kf_init)
o_T_b_cf_init = pin.SE3(o_R_b_cf_init, o_p_ob_cf_init)

w_T_okf = w_T_m_init * o_T_b_kf_init.inverse()
w_T_ocf = w_T_m_init * o_T_b_cf_init.inverse()

# transform estimated trajectories in mocap frame
w_p_wb_kf_arr = np.array([w_T_okf.act(o_p_ob_kf) for o_p_ob_kf in o_p_ob_kf_arr])
w_p_wb_cf_arr = np.array([w_T_ocf.act(o_p_ob_cf) for o_p_ob_cf in o_p_ob_cf_arr])

# TODO: same for orientations

#####################################
# MOCAP freq 200Hz vs robot freq 1kHz
NB_OVER = 5
# compute filterd Mo-Cap velocity
w_p_wm_arr_sub = w_p_wm_arr[::NB_OVER]
w_v_wm_arr_sagol_sub = signal.savgol_filter(w_p_wm_arr_sub, window_length=21, polyorder=3, deriv=1, axis=0, delta=NB_OVER*dt, mode='mirror')
w_v_wm_arr_sagol = w_v_wm_arr_sagol_sub.repeat(NB_OVER, axis=0) 
# compute mocap base velocities
w_R_m_lst = [pin.Quaternion(w_q_m.reshape((4,1))).toRotationMatrix() for w_q_m in w_q_m_arr] 
m_v_wm_arr_filt = np.array([w_R_m.T @ w_v_wm for w_R_m, w_v_wm in zip(w_R_m_lst, w_v_wm_arr_sagol)])
print(m_v_wm_arr_filt.shape)


# Create a continuous norm to map from data points to colors
def plot_xy_traj(xy, cmap, fig, ax, cstart):
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t_arr[0], t_arr[-1])
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(t_arr)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)
    plt.plot(xy[0,0], xy[0,1], cstart+'x')


fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
plot_xy_traj(w_p_wm_arr[:,:2], 'winter', fig, ax, 'b')
plot_xy_traj(w_p_wb_cf_arr[:,:2], 'autumn', fig, ax, 'r')
plot_xy_traj(w_p_wb_kf_arr[:,:2], 'spring', fig, ax, 'r')
fig.tight_layout()

xmin = w_p_wm_arr[:,0].min() 
xmax = w_p_wm_arr[:,0].max() 
ymin = w_p_wm_arr[:,1].min() 
ymax = w_p_wm_arr[:,1].max()

xmin = min(xmin, w_p_wb_cf_arr[:,0].min()) 
xmax = max(xmax, w_p_wb_cf_arr[:,0].max()) 
ymin = min(ymin, w_p_wb_cf_arr[:,1].min()) 
ymax = max(ymax, w_p_wb_cf_arr[:,1].max())

xmin = min(xmin, w_p_wb_kf_arr[:,0].min()) 
xmax = max(xmax, w_p_wb_kf_arr[:,0].max()) 
ymin = min(ymin, w_p_wb_kf_arr[:,1].min()) 
ymax = max(ymax, w_p_wb_kf_arr[:,1].max())

offx = 0.1*(xmax - xmin)
offy = 0.1*(ymax - ymin)

# line collections don't auto-scale the plot
plt.xlim(xmin-offx, xmax+offx) 
plt.ylim(ymin-offy, ymax+offy)
plt.grid()





# # Compute orientation as roll pitch yaw
# R_es_aligned = [pin.Quaternion(q.reshape((4,1))).toRotationMatrix() for q in traj.q_es_aligned]
# R_gt = [pin.Quaternion(q.reshape((4,1))).toRotationMatrix() for q in traj.q_gt]
# rpy_es_aligned = np.array([pin.rpy.matrixToRpy(R) for R in R_es_aligned])
# rpy_gt = np.array([pin.rpy.matrixToRpy(R) for R in R_gt])



# PLOT parameters
# FIGSIZE = (3.14,2.8)
FIGSIZE = (6,5)
GRID = True
# EXT = '.png'
EXT = '.pdf'



# fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
# fig.canvas.set_window_title('base_orientation_base_frame')
# ylabels = ['Roll [rad]', 'Pitch [rad]', 'Yaw [rad]']
# for i in range(3):
#     axs[i].plot(t_arr, rpy_es_aligned[:,i], 'b', markersize=1, label='est')
#     axs[i].plot(t_arr, rpy_gt[:,i], 'r', markersize=1, label='Mo-Cap')
#     axs[i].set_ylabel(ylabels[i])
#     axs[i].yaxis.set_label_position("right")
#     axs[i].grid(GRID)
# axs[2].set_xlabel('time [s]')
# axs[0].legend()

fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
fig.canvas.set_window_title('base_velocity_base_frame')
axs[0].set_title('base_velocity_base_frame')
ylabels = ['Vx [m/s]', 'Vy [m/s]', 'Vz [m/s]']
for i in range(3):
    axs[i].plot(t_arr, b_v_ob_kf_arr[:,i], markersize=1, label='KF')
    axs[i].plot(t_arr, b_v_ob_cf_arr[:,i], markersize=1, label='CF')
    axs[i].plot(t_arr, m_v_wm_arr_filt[:,i], markersize=1, label='Mo-Cap')
    axs[i].set_ylabel(ylabels[i])
    axs[i].yaxis.set_label_position("right")
    axs[i].grid(GRID)
axs[2].set_xlabel('time [s]')
axs[0].legend()

fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
fig.canvas.set_window_title('base_position')
axs[0].set_title('base_position')
ylabels = ['Px [m]', 'Py [m]', 'Pz [m]']
for i in range(3):
    axs[i].plot(t_arr, w_p_wb_kf_arr[:,i], markersize=1, label='KF')
    axs[i].plot(t_arr, w_p_wb_cf_arr[:,i], markersize=1, label='CF')
    axs[i].plot(t_arr, w_p_wm_arr[:,i], markersize=1, label='Mo-Cap')
    axs[i].set_ylabel(ylabels[i])
    axs[i].yaxis.set_label_position("right")
    axs[i].grid(GRID)
axs[2].set_xlabel('time [s]')
axs[0].legend()



plt.show()
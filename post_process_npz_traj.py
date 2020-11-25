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
print(cwdir)
DATA_FOLDER_RESULTS = os.path.join(cwdir, 'data/quadruped_experiments_results/')
print(DATA_FOLDER_RESULTS)

# from wolf estimation
data_file = 'out.npz'
data_file_post = 'out_post.npz'

# Keys:
print('Reading ', DATA_FOLDER_RESULTS+data_file)
arr_dic = np.load(DATA_FOLDER_RESULTS+data_file)
arr_dic = shortened_arr_dic(arr_dic, 0, 10000)

dt = 1e-3
t_arr = arr_dic['t']
N = len(t_arr)
print('N: ', N)
# GT
w_p_wm_arr = arr_dic['w_p_wm']
w_q_m_arr = arr_dic['w_q_m']
w_v_wm_arr = arr_dic['w_v_wm']
m_v_wm_arr = arr_dic['m_v_wm']

q_kf_arr = arr_dic['q_kf']
v_kf_arr = arr_dic['v_kf']
q_cf_arr = arr_dic['q_cf']
v_cf_arr = arr_dic['v_cf']

b_v_ob_kf_arr = v_kf_arr[:,:3]
b_v_ob_cf_arr = v_cf_arr[:,:3]



#####################################
# MOCAP freq 200Hz vs robot freq 1kHz
NB_OVER = 5

# compute filterd Mo-Cap velocity
w_p_wm_arr_sub = w_p_wm_arr[::NB_OVER]
w_v_wm_arr_sagol_sub = signal.savgol_filter(w_p_wm_arr_sub, window_length=21, polyorder=3, deriv=1, axis=0, delta=NB_OVER*dt, mode='mirror')
w_v_wm_arr_sagol = w_v_wm_arr_sagol_sub.repeat(NB_OVER, axis=0) 

# compute base velocities
w_R_m_lst = [pin.Quaternion(w_q_m.reshape((4,1))).toRotationMatrix() for w_q_m in w_q_m_arr] 
m_v_wm_arr_filt = np.array([w_R_m.T @ w_v_wm for w_R_m, w_v_wm in zip(w_R_m_lst, w_v_wm_arr_sagol)])


# #####################
# # Trajectory alignment
# pose_est = np.hstack([t_arr.reshape((N,1)), o_p_ob_arr, o_q_b_arr])
# pose_gtr = np.hstack([t_arr.reshape((N,1)), w_p_wm_arr, w_q_m_arr])

# # filter out end of traj where the robot solo causes strange jump in Mo-Cap position
# # NPREV = 12
# # t_arr = t_arr[:N-NPREV] 
# # pose_est = pose_est[:N-NPREV,:]
# # pose_gtr = pose_gtr[:N-NPREV,:]
# # m_v_wm_arr_filt = m_v_wm_arr_filt[:N-NPREV,:]
# # b_v_ob_arr = b_v_ob_arr[:N-NPREV,:]
# # b_v_oc_arr = b_v_oc_arr[:N-NPREV,:]

# res_folder = 'res_for_rpg_'+datetime.datetime.now().strftime("%y_%m_%d__%H_%M_%S") + '/'
# os.makedirs(res_folder)
# np.savetxt(res_folder+'stamped_traj_estimate.txt', pose_est, delimiter=' ')
# np.savetxt(res_folder+'stamped_groundtruth.txt',   pose_gtr, delimiter=' ')

# traj = rpg_traj.Trajectory(res_folder, align_type='se3', align_num_frames=1)  # settings ensured by eval_cfg.yaml
# # 'a' like aligned
# a_p_ab_arr = traj.p_es_aligned
# a_q_b_arr = traj.q_es_aligned

# # compute the relative transform applied to the estimation trajectory and propagate it to center of mass quantities

# a_T_b0 = pin.SE3(pin.Quaternion(a_q_b_arr[0,:].reshape((4,1))).toRotationMatrix(), a_p_ab_arr[0,:])
# o_T_b0 = pin.SE3(pin.Quaternion(o_q_b_arr[0,:].reshape((4,1))).toRotationMatrix(), o_p_ob_arr[0,:])
# a_T_o = a_T_b0 * o_T_b0.inverse()  # compute alignment transformation based on the first frame
# a_R_o = a_T_o.rotation

# # align CoM trajectory
# a_p_ac_arr = [a_T_o*o_p_oc for o_p_oc in o_p_oc_arr]
# a_v_ac_arr = [a_R_o*o_v_oc for o_v_oc in o_v_oc_arr]
# a_L_arr = [a_R_o@o_Lc for o_Lc in o_Lc_arr]


# # save in a new file all needed things for display
# keys_to_keep = ['t', 'w_p_wm', 'w_q_m', 'qa']
# arr_dic_post = {k: arr_dic[k] for k in keys_to_keep}
# arr_dic_post['a_p_ab'] = a_p_ab_arr
# arr_dic_post['a_q_b'] = a_q_b_arr
# arr_dic_post['a_p_ac'] = a_p_ac_arr 
# arr_dic_post['a_L'] = a_L_arr 

# print('Saving ', DATA_FOLDER_RESULTS+data_file_post)
# np.savez(DATA_FOLDER_RESULTS+data_file_post, **arr_dic_post)



# traj.compute_absolute_error()
# traj.compute_boxplot_distances()
# traj.compute_relative_errors()
# traj.compute_relative_error_at_subtraj_len()

# Create a continuous norm to map from data points to colors

# def plot_xy_traj(xy, cmap, fig, ax, cstart):
#     points = xy.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     norm = plt.Normalize(t_arr[0], t_arr[-1])
#     lc = LineCollection(segments, cmap=cmap, norm=norm)
#     # Set the values used for colormapping
#     lc.set_array(t_arr)
#     lc.set_linewidth(2)
#     line = ax.add_collection(lc)
#     # fig.colorbar(line, ax=ax)
#     plt.plot(xy[0,0], xy[0,1], cstart+'x')


# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# plot_xy_traj(traj.p_es_aligned[:,:2], 'autumn', fig, ax, 'b')
# plot_xy_traj(traj.p_gt[:,:2], 'winter', fig, ax, 'r')
# fig.tight_layout()

# xmin = traj.p_es_aligned[:,0].min() 
# xmax = traj.p_es_aligned[:,0].max() 
# ymin = traj.p_es_aligned[:,1].min() 
# ymax = traj.p_es_aligned[:,1].max()

# xmin = min(xmin, traj.p_gt[:,0].min()) 
# xmax = max(xmax, traj.p_gt[:,0].max()) 
# ymin = min(ymin, traj.p_gt[:,1].min()) 
# ymax = max(ymax, traj.p_gt[:,1].max())

# offx = 0.1*(xmax - xmin)
# offy = 0.1*(ymax - ymin)

# # line collections don't auto-scale the plot
# plt.xlim(xmin-offx, xmax+offx) 
# plt.ylim(ymin-offy, ymax+offy)
# plt.grid()





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

# fig, axs = plt.subplots(3,1, figsize=FIGSIZE)
# fig.canvas.set_window_title('base_position')
# ylabels = ['Px [m]', 'Py [m]', 'Pz [m]']
# for i in range(3):
#     axs[i].plot(t_arr, traj.p_es_aligned[:,i], 'b', markersize=1, label='est')
#     axs[i].plot(t_arr, traj.p_gt[:,i], 'r', markersize=1, label='Mo-Cap')
#     axs[i].set_ylabel(ylabels[i])
#     axs[i].yaxis.set_label_position("right")
#     axs[i].grid(GRID)
# axs[2].set_xlabel('time [s]')
# axs[0].legend()



plt.show()
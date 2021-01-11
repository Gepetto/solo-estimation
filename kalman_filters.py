from functools import partial
import numpy as np
import pinocchio as pin
from example_robot_data import load

def screw(v):
    return np.array([0,   -v[2],  v[1],
                     v[1],   0,  -v[0],
                    -v[1], v[0],  0    ]).reshape((3,3))

def cross3(u, v):
    # np.cross > 10 times slower for 1d arrays 
    return np.array([
            u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0]
            ])

def q2R(q_arr):
    return pin.Quaternion(q_arr.reshape((4,1))).toRotationMatrix()


class ImuLegKF:

    def __init__(self, dt, q_init):
        self.robot = load('solo12')

        # stores a look up table for contact state position
        LEGS = ['FL', 'FR', 'HL', 'HR']
        self.contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
        self.contact_ids = [self.robot.model.getFrameId(leg_name) for leg_name in self.contact_frame_names]
        self.nbc = len(self.contact_ids)

        # position of contact of id "cid" in state vector
        self.cid_stateidx_map = {cid: 6+3*i for i, cid in enumerate(self.contact_ids)}

        # Base to IMU transformation
        self.b_p_bi = np.zeros(3)
        # self.b_p_bi = np.array([0.1163, 0.0, 0.02])
        b_q_i  = np.array([0, 0, 0, 1])

        self.b_T_i = pin.SE3(q2R(b_q_i), self.b_p_bi)
        self.i_T_b = self.b_T_i.inverse()
        self.b_R_i = self.b_T_i.rotation
        self.i_R_b = self.i_T_b.rotation
        self.i_p_ib = self.i_T_b.translation

        # state structure: [base position, base velocity, [feet position] ] 
        # all quantities expressed in universe frame
        # [o_p_ob, o_v_o1, o_p_ol1, o_p_ol1, o_p_ol1, o_p_ol3]
        o_p_ol_lst = [self.robot.framePlacement(q_init, leg_id).translation for leg_id in self.contact_ids]
        # for o_p_ol in o_p_ol_lst: 
        #     print(o_p_ol)
        # initialize the state
        o_p_ob = q_init[0:3]
        o_R_i = q2R(q_init[3:7])
        o_p_oi = o_p_ob - o_R_i@self.i_p_ib
        self.x = np.concatenate((o_p_oi, np.zeros(3), *o_p_ol_lst))
        self.state_size = 6+self.nbc*3

        # state prior cov
        # std_p_prior = 0.01*np.ones(3)
        std_p_prior = np.array([0.0001, 0.0001, 0.1])
        std_v_prior = 1*np.ones(3)
        std_pl_priors = 0.1*np.ones(3*self.nbc)
        std_prior = np.concatenate((std_p_prior, std_v_prior, std_pl_priors))
        self.P = np.diag(std_prior)**2

        # discretization period
        self.dt = dt

        # filter noises
        std_kf_dic = {
            'std_foot': 0.005,   # m/sqrt(Hz) process noise on foot dynamics when in contact -> raised when stable contact interuption
            'std_acc': 0.08,    # noise on linear acceleration measurements (m.s-2), add a slack to account for bias
            'std_wb': 0.01,    # noise on angular velocity measurements (rad.s-1), add a slack to account for bias 
            'std_qa': 0.05,     # noise on joint position measurements (rad)
            'std_dqa': 5,    # noise on joint velocity measurements (rad.s-1)
            'std_kin': 0.005,    # kinematic uncertainties (m)
            'std_hfoot': 0.001, # terrain height uncertainty -> roughness of the terrain
        } 

        # static jacs and covs
        self.Qfoot = std_kf_dic['std_foot']**2 * np.eye(3)
        self.Qacc = std_kf_dic['std_acc']**2 * np.eye(3)
        self.Qwb = std_kf_dic['std_wb']**2 * np.eye(3)
        self.Qqa = std_kf_dic['std_qa']**2 * np.eye(self.robot.model.nv - 6)
        self.Qdqa = std_kf_dic['std_dqa']**2 * np.eye(self.robot.model.nv - 6)
        self.Qkin = std_kf_dic['std_kin']**2 * np.eye(3)
        self.Qhfoot = std_kf_dic['std_hfoot']**2
        
        # propagation
        self.Fk = self.propagation_jac()

        self.H_vel = self.vel_meas_H()
        self.H_relp_dic = {cid: self.relp_meas_H(cid_idx) for cid, cid_idx in self.cid_stateidx_map.items()}

        # measurements to be used in KF update by default
        # self.meas_choices = (0,0,0)  # nothing happens
        # self.meas_choices = (1,0,0)  # only relp
        # self.meas_choices = (0,1,0)  # only diff kin
        # self.meas_choices = (1,1,0)  # relp + diff kin
        # self.meas_choices = (1,0,1)  # relp + foot height
        self.meas_choices = (1,1,1)  # all kinematics + foot height

        # contact detection
        self.k_since_contact = np.zeros(4)


        self.o_v_oi_dic = {i: [] for i in range(4)}

    def run_filter(self, o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status):
        self.update_data(o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status)
        self.detect_feets_in_contact()
        self.propagate()
        self.correct()
    
    def update_data(self, o_a_oi, o_R_i, qa, dqa, i_omg_oi, contact_status):
        self.o_a_oi = o_a_oi   
        self.o_R_i = o_R_i   
        self.qa = qa   
        self.dqa = dqa   
        self.i_omg_oi = i_omg_oi 
        self.contact_status = contact_status

        self.o_R_b = o_R_i@self.i_R_b

    def get_state(self):
        return self.x
        
    def get_configurations(self):
        # return state as configuration arrays
        self.update_base_state()
        q = np.concatenate([self.o_p_ob, self.o_q_b, self.qa])
        v = np.concatenate([self.b_v_ob, self.b_omg_ob, self.dqa])
        return q, v

    def update_base_state(self):
        # state estimation is done in IMU frame -> composition to base frame
        self.o_R_b = self.o_R_i @ self.i_R_b
        self.o_q_b = pin.Quaternion(self.o_R_b).coeffs()
        self.o_p_ob = self.x[0:3] + self.o_R_i @ self.i_p_ib
        self.o_v_ob = self.x[3:6] + self.o_R_i @ cross3(self.i_omg_oi, self.i_p_ib)
        self.b_omg_ob = self.b_R_i@self.i_omg_oi

        # from current estimates
        self.b_v_ob = self.o_R_b.T@self.o_v_ob

        return 0

    def detect_feets_in_contact(self):
        self.k_since_contact += self.contact_status  # Increment feet in stance phase
        self.k_since_contact *= self.contact_status  # Reset feet in swing phase
        self.feets_in_contact = (self.k_since_contact > 50) 

    def propagate(self):
        """
        o_a_oi: acceleration of the imu wrt world frame in world frame (o_R_i*imu_acc + o_g)
        (IMU proper acceleration rotated in world frame)
        feets_in_contact: options
             - size 4 boolean array
             - 
        """
        # state propagation
        self.x[0:3] += self.x[3:6] * self.dt + 0.5*self.o_a_oi*self.dt**2
        self.x[3:6] += self.o_a_oi*self.dt
        
        # cov propagation
        # adjust Qk depending on feet in contact?
        self.propagate_cov(self.feets_in_contact)

    def correct(self, meas_choices=None):
        """
        measurements: which measurements to use -> geometry?, differential?, zero height?
        """
        if meas_choices is None:
            meas_choices = self.meas_choices
        # just to be sure we do not have base placement/velocity in q and dq
        q_st = np.concatenate([np.array(6*[0]+[1]), self.qa])
        dq_st = np.concatenate([np.zeros(6), self.dqa])
        # update the robot state, freeflyer at the universe not moving
        self.robot.forwardKinematics(q_st, dq_st)
        #################
        # Geometry update
        #################
        # for each foot (in contact or not) update position using kinematics
        # Adapt cov if in contact or not
        
        if meas_choices[0]:
            self.correct_relp(q_st, self.o_R_i,  self.feets_in_contact)

        ################################
        # differential kinematics update
        ################################
        # For feet in contact only, use the zero velocity assumption to derive base velocity measures
        if meas_choices[1]:
            self.correct_relv(q_st, dq_st, self.i_omg_oi, self.o_R_i, self.feets_in_contact)

        # ####################################
        # # foot in contact zero height update
        # ####################################
        if meas_choices[2]:
            self.correct_height(self.feets_in_contact)
    
    def correct_relp(self, q_st, o_R_i,  feets_in_contact):
        for i_ee, cid in enumerate(self.contact_ids):
            if feets_in_contact[i_ee]:
                b_p_bl = self.robot.framePlacement(q_st, cid, update_kinematics=True).translation
                i_p_il = self.i_T_b * b_p_bl 
                o_p_il = o_R_i @ i_p_il
                Q_relp = self.relp_cov(q_st, o_R_i, cid)
                # if feets_in_contact[i_ee]:
                #     Q_relp *= 10  # crank up covariance: foot rel position less reliable when in air (really?)
                self.kalman_update(o_p_il, self.H_relp_dic[cid], Q_relp)

    def correct_relv(self, q_st, dq_st, i_omg_oi, o_R_i, feets_in_contact):
        for i_ee, cid in enumerate(self.contact_ids):
            if feets_in_contact[i_ee]:
                # measurement: velocity in world frame
                b_T_l = self.robot.framePlacement(q_st, cid, update_kinematics=False)
                b_p_bl = b_T_l.translation
                i_p_il = self.i_T_b * b_p_bl
                b_R_l = b_T_l.rotation

                # retrieve relative foot base velocity
                l_v_bl = self.robot.frameVelocity(q_st, dq_st, cid, update_kinematics=False).linear

                # measurement: velocity in world frame
                # print(np.linalg.norm(l_v_bl)/np.linalg.norm(cross3(i_omg_oi, i_p_il)))
                o_v_oi = - self.o_R_b @ b_R_l @ l_v_bl - self.o_R_i @ cross3(i_omg_oi, i_p_il)
                Q_vel = self.vel_meas_cov(q_st, i_omg_oi, o_R_i, cid)

                self.o_v_oi_dic[i_ee].append(o_v_oi)

                self.kalman_update(o_v_oi, self.H_vel, Q_vel)
            else:
                self.o_v_oi_dic[i_ee].append(np.zeros(3))

    def correct_height(self, feets_in_contact):
        for i_ee, cid in enumerate(self.contact_ids):
            if feets_in_contact[i_ee]:        
                # zero height update
                hfoot = 0.0
                H = np.zeros(self.state_size)
                H[self.cid_stateidx_map[cid]+2] = 1
                self.kalman_update(hfoot, H, self.Qhfoot)
                
    def propagation_jac(self):
        F = np.eye(self.x.shape[0])
        F[0:3,3:6] = self.dt*np.eye(3)
        return F

    def propagate_cov(self, feets_in_contact):
        # state propagation
        self.P = self.Fk @ self.P @ self.Fk.T

        # add noise from accelerometer sampling
        self.P[3:6,3:6] += self.Qacc * self.dt**2

        # feet perturbation integration -> if not in contact add big covariance
        for i, i_fi in enumerate(self.cid_stateidx_map.values()):
            if feets_in_contact[i]:
                self.P[i_fi:i_fi+3,i_fi:i_fi+3] += self.Qfoot*self.dt
            else:
                self.P[i_fi:i_fi+3,i_fi:i_fi+3] += 100*np.eye(self.Qfoot.shape[0])*self.dt

    def relp_meas_H(self, cid_idx):
        H = np.zeros((3,self.state_size))
        H[:3,:3] = - np.eye(3)
        H[:3,cid_idx:cid_idx+3] = np.eye(3) 
        return H

    def vel_meas_H(self):
        H = np.zeros((3, self.state_size))
        H[0:3,3:6] = np.eye(3)
        return H

    def relp_cov(self, q_st, o_R_i, cid):
        bTl = self.robot.framePlacement(q_st, cid, update_kinematics=False)
        o_Jqa_i = o_R_i @ bTl.rotation @ self.robot.computeFrameJacobian(q_st, cid)[:3,6:]
        return o_Jqa_i @ self.Qqa @ o_Jqa_i.T + self.Qkin
    
    def vel_meas_cov(self, q_st, wb, o_R_i, cid):
        wbx = screw(wb)
        bTl = self.robot.framePlacement(q_st, cid, update_kinematics=False)
        b_Jl = bTl.rotation @ self.robot.computeFrameJacobian(q_st, cid)[:3,6:]
        b_p_bl_x = screw(bTl.translation)
        # minuses due to relation [.]x^T = -[.]x
        return o_R_i@(b_Jl @ self.Qdqa @ b_Jl.T - b_p_bl_x @ self.Qwb @ b_p_bl_x - wbx @ b_Jl @ self.Qqa @ b_Jl.T @ wbx)@o_R_i.T
    
    def kalman_update(self, y, H, R):
        # general unoptimized kalman update
        # innovation z = y - h(x)
        # Innov cov Z = HPH’ + R ; avec H = dh/dx = - dz/dx
        # Kalman gain K = PH’ / Z
        # state error dx = K*z
        # State update x <-- x (+) dx
        # Cov update P <-- P - KZP

        z = y - H @ self.x
        Z = H @ self.P @ H.T + R
        if isinstance(Z, np.ndarray):
            K = self.P @ H.T @ np.linalg.inv(Z)
            dx = K @ z
        else:  # for scalar measurements
            K = self.P @ H.T / Z
            dx = K * z
            # reshape to avoid confusion between inner and outer product when multiplying 2 1d arrays
            K = K.reshape((self.state_size,1))
            H = H.reshape((1,self.state_size))
        self.x = self.x + dx
        self.P = self.P - K @ H @ self.P
        
        return z, dx

def base_vel_from_stable_contact(robot, q, dq, i_omg_oi, o_R_i, cid):
    """
    Assumes forwardKinematics has been called on the robot object with current q dq
    And that q = [0,0,0, 0,0,0,1, qa]
            dq = [0,0,0, 0,0,0, dqa]
    """
    b_T_l = robot.framePlacement(q, cid, update_kinematics=False)
    b_p_bl = b_T_l.translation
    b_R_l = b_T_l.rotation

    l_v_bl = robot.frameVelocity(q, dq, cid, update_kinematics=False).linear
    # measurement: velocity in world frame
    b_v_ob = - b_R_l @ l_v_bl + cross3(b_p_bl, i_omg_oi)
    return o_R_i @ b_v_ob

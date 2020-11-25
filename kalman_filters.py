from functools import partial
import numpy as np
import pinocchio as pin
from example_robot_data import load
from contact_forces_estimator import ContactForcesEstimator

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

class ImuLegKF:

    def __init__(self, dt, q_init):
        self.robot = load('solo12')

        # stores a look up table for contact state position
        LEGS = ['FL', 'FR', 'HL', 'HR']
        contact_frame_names = [leg+'_ANKLE' for leg in LEGS]
        self.contact_ids = [self.robot.model.getFrameId(leg_name) for leg_name in contact_frame_names]
        nbc = len(self.contact_ids)

        # position of contact of id "cid" in state vector
        self.cid_stateidx_map = {cid: 6+3*i for i, cid in enumerate(self.contact_ids)}

        # state structure: [base position, base velocity, [feet position] ] 
        # all quantities expressed in universe frame
        # [o_p_ob, o_v_o1, o_p_ol1, o_p_ol1, o_p_ol1, o_p_ol3]
        o_p_ol_lst = [self.robot.framePlacement(q_init, leg_id).translation for leg_id in self.contact_ids]
        # initialize the state
        self.x = np.concatenate((q_init[0:3], np.zeros(3), *o_p_ol_lst))
        self.state_size = 6+nbc*3

        # Base to IMU transformation
        # b_p_bi = np.zeros(3)
        self.b_p_bi = np.array([0.1163, 0.0, 0.02])
        b_q_i  = np.array([0, 0, 0, 1])

        self.b_T_i = pin.SE3((pin.Quaternion(b_q_i.reshape((4,1)))).toRotationMatrix(), self.b_p_bi)
        self.i_T_b = self.b_T_i.inverse()
        self.b_R_i = self.b_T_i.rotation
        self.i_R_b = self.i_T_b.rotation
        self.i_p_ib = self.i_T_b.translation

        # state prior cov
        std_p_prior = 0.01*np.ones(3)
        std_v_prior = 1*np.ones(3)
        std_pl_priors = 10*np.ones(3*nbc)
        std_prior = np.concatenate((std_p_prior, std_v_prior, std_pl_priors))
        self.P = np.diag(std_prior)**2

        # discretization period
        self.dt = dt

        # filter noises
        std_kf_dic = {
            'std_foot': 0.1,  # m/sqrt(Hz) process noise on foot dynamics when in contact -> raised when stable contact interuption
            'std_acc': 0.1,# noise on linear acceleration measurements
            'std_wb': 0.001,# noise on angular velocity measurements
            'std_qa': 0.05,# noise on joint position measurements
            'std_dqa': 0.05,# noise on joint velocity measurements
            'std_kin': 0.01,# kinematic uncertainties
            'std_hfoot': 0.01,# terrain height uncertainty -> roughness of the terrain
        } 

        # static jacs and covs
        self.Qacc = std_kf_dic['std_acc']**2 * np.eye(3)
        self.Qwb = std_kf_dic['std_wb']**2 * np.eye(3)
        self.Qqa = std_kf_dic['std_qa']**2 * np.eye(self.robot.model.nv - 6)
        self.Qdqa = std_kf_dic['std_dqa']**2 * np.eye(self.robot.model.nv - 6)
        self.Qfoot = std_kf_dic['std_foot']**2 * np.eye(3)
        self.Qkin = std_kf_dic['std_kin']**2 * np.eye(3)
        self.Qhfoot = std_kf_dic['std_hfoot']**2
        
        # propagation
        self.Fk = self.propagation_jac()

        self.H_vel = self.vel_meas_H()
        self.H_relp_dic = {cid: self.relp_meas_H(cid_idx) for cid, cid_idx in self.cid_stateidx_map.items()}

        # measurements to be used in KF update by default
        # self.measurements = (0,0,0)  # nothing happens
        # self.measurements = (1,0,0)  # only kin
        # self.measurements = (0,1,0)  # only diff kin
        self.measurements = (1,1,0)  # all kinematics
        # self.measurements = (1,1,1)  # all kinematics + foot height

        # contact detection
        self.k_since_contact = np.zeros(4)

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

    def get_state_base(self):
        # state estimation is done in IMU frame -> composition to base frame
        self.o_R_b = self.o_R_i @ self.i_R_b
        self.o_q_b = pin.Quaternion(self.o_R_b).coeffs()
        self.o_p_ob = self.x[0:3] + self.o_R_i @ self.i_p_ib
        self.o_v_ob = self.x[3:6] + self.o_R_i @ cross3(self.i_omg_oi, self.i_p_ib)
        self.b_omg_ob = self.b_R_i@self.i_omg_oi

        # from current estimates
        self.b_v_ob = self.o_R_b.T@self.o_v_ob

        # only for force estimation
        self.o_a_ob = self.o_a_oi + self.o_R_i@cross3(self.i_omg_oi, cross3(self.i_omg_oi, self.i_p_ib))     # acceleration composition (neglecting i_domgdt_oi x i_p_ib)

        return 0
    
    def get_configurations(self):
        # return state as configuration arrays
        self.get_state_base()
        q = np.concatenate([self.o_p_ob, self.o_q_b, self.qa])
        v = np.concatenate([self.o_R_b.T@self.o_v_ob, self.b_omg_ob, self.dqa])
        return q, v

    def detect_feets_in_contact(self):
        self.k_since_contact += self.contact_status  # Increment feet in stance phase
        self.k_since_contact *= self.contact_status  # Reset feet in swing phase
        self.feets_in_contact = (self.k_since_contact > 16) 

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

    def correct(self, measurements=None):
        """
        measurements: which measurements to use -> geometry?, differential?, zero height?
        """
        if measurements is None:
            measurements = self.measurements
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
        
        if measurements[0]:
            self.correct_relp(q_st, self.o_R_i,  self.feets_in_contact)

        ################################
        # differential kinematics update
        ################################
        # For feet in contact only, use the zero velocity assumption to derive base velocity measures
        if measurements[1]:
            self.correct_relv(q_st, dq_st, self.i_omg_oi, self.o_R_i, self.feets_in_contact)

        # ####################################
        # # foot in contact zero height update
        # ####################################
        if measurements[2]:
            self.correct_height(self.feets_in_contact)
    
    def correct_relp(self, q_st, o_R_i,  feets_in_contact):
        for i_ee, cid in enumerate(self.cid_stateidx_map):
            b_p_bl = self.robot.framePlacement(q_st, cid, update_kinematics=False).translation
            i_p_il =  self.i_T_b * b_p_bl 
            o_p_il = o_R_i @ i_p_il
            Q_relp = self.relp_cov(q_st, o_R_i, cid)
            # if feets_in_contact[i_ee]:
            #     Q_relp *= 10  # crank up covariance: foot rel position less reliable when in air (really?)
            self.kalman_update(o_p_il, self.H_relp_dic[cid], Q_relp)

    def correct_relv(self, q_st, dq_st, i_omg_oi, o_R_i, feets_in_contact):
        for i_ee, cid in enumerate(self.contact_ids):
            if feets_in_contact[i_ee]:
                # measurement: velocity in world frame
                o_v_ob = base_vel_from_stable_contact(self.robot, q_st, dq_st, i_omg_oi, o_R_i, cid)
                o_v_oi = o_v_ob + o_R_i @ cross3(i_omg_oi, self.i_R_b@self.b_p_bi)  # velocity composition law

                Q_vel = self.vel_meas_cov(q_st, i_omg_oi, o_R_i, cid)

                self.kalman_update(o_v_ob, self.H_vel, Q_vel)

    def correct_height(self, feets_in_contact):
        for i_ee, cid in enumerate(self.contact_ids):
            if feets_in_contact[i_ee]:        
                # zero height update
                hfoot = 0.0
                H = np.zeros(self.state_size)
                H[self.cid_stateidx_map[cid]+2] = 1
                self.kalman_update(hfoot, H, self.Qhfoot)

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
                self.P[i_fi:i_fi+3,i_fi:i_fi+3] += 10000*self.Qfoot*self.dt

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
 

def base_vel_from_stable_contact(robot, q, dq, i_omg_oi, o_R_i, cid):
    """
    Assumes forwardKinematics has been called on the robot object with current q dq
    And that q = [0,0,0, 0,0,0,1, qa]
            dq = [0,0,0, 0,0,0, dqa]
    """
    b_T_l = robot.framePlacement(q, cid, update_kinematics=False)
    b_p_bl = b_T_l.translation
    bRl = b_T_l.rotation

    l_v_bl = robot.frameVelocity(q, dq, cid, update_kinematics=False).linear
    # measurement: velocity in world frame
    b_v_ob = - bRl @ l_v_bl + cross3(b_p_bl, i_omg_oi)
    return o_R_i @ b_v_ob
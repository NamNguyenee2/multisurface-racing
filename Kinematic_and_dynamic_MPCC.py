# MIT License
# MPCC Implementation - Converted from ST-MPC
# Key changes: 
# - State augmented with theta (arc length) and v_theta (progress velocity)
# - Contouring error (lateral deviation) + Lag error (longitudinal deviation)
# - Progress maximization term

import time
from dataclasses import dataclass, field
import cvxpy
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
import copy
from scipy.interpolate import make_interp_spline, CubicSpline

class TrackRef_MPCC:
    def __init__(self, waypoints):
        # Extract X and Y coordinates
        if np.linalg.norm(waypoints[0] - waypoints[-1]) > 200:
            waypoints = np.vstack([waypoints, waypoints[0]])
        self.x = waypoints[:, 1]

        self.y = waypoints[:, 2]

        self.track_length = len(self.x)

        self.theta_arr = np.linspace(0, self.track_length, num=len(self.x))
        self.spline_x = CubicSpline(self.theta_arr, self.x, bc_type='periodic')
        self.spline_y = CubicSpline(self.theta_arr, self.y, bc_type='periodic')

    def _wrap_theta(self, theta):
        return theta % self.track_length

    def get_position(self, theta):
        theta = self._wrap_theta(theta)
        x = float(self.spline_x(theta))
        y = float(self.spline_y(theta))
        return x, y

    def get_phi(self, theta):
        theta = self._wrap_theta(theta)
        dx = self.spline_x(theta, 1)
        dy = self.spline_y(theta, 1)
        return float(np.arctan2(dy, dx))

class STMPCCPlanner:
    """
    Model Predictive Contouring Control (MPCC) Planner
    
    State vector (already includes theta):
    x = [x, y, v, phi, vx, vy, omega, theta]  (NXK = 8)
    
    Control input (now includes v_theta):
    u = [acceleration, steering_angle, v_theta]  (NU = 3)
    
    Where:
    - theta: arc length position on track [0, track_length] (STATE)
    - v_theta: rate of progress along track (CONTROL INPUT)
    """

    def __init__(self, model, config, waypoints=None, track=None):
        self.waypoints = waypoints
        self.model = model
        self.config = config
        self.track = track
        
        # Initialize track reference system
        self.TrackRef = TrackRef_MPCC(self.waypoints)
        self.track_length = self.TrackRef.track_length
        
        # Assume theta is the last state (index -1)
        self.theta_index = self.config.NXK - 1
        
        # MPC state variables (theta already included in NXK)
        self.input_o = np.zeros(self.config.NU) * np.NAN  # Now NU = 3 (accel, steering, v_theta)
        self.states_output = np.ones((self.config.NXK, self.config.TK +1)) * np.NaN
        
        # MPCC weights
        self.q_contour = self.config.q_contour      # Contouring error weight
        self.q_lag = self.config.q_lag           # Lag error weight  
        self.q_theta = self.config.q_theta        # Progress maximization (negative = reward)
        
        self.mpc_prob_init()


    def plan(self, states, waypoints=None):
        """
        Main planning function
        """
        if waypoints is not None:
            self.waypoints = waypoints
            self.TrackRef = TrackRef_MPCC(self.waypoints)
            self.track_length = self.TrackRef.track_length

        u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = self.MPCC_Control(states)
        return u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy

    def get_reference_trajectory_mpcc(self, theta_array):
        ref_traj = np.zeros((self.config.NXK + 2, self.config.TK + 1))
        
        for i, theta in enumerate(theta_array):
            theta_wrapped = theta % self.track_length
            # Reference position
            x_ref, y_ref = self.TrackRef.get_position(theta_wrapped)
            ref_traj[0, i] = x_ref
            ref_traj[1, i] = y_ref
      
        return ref_traj
    
    def mpc_prob_init(self):
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))  # Now includes v_theta
        
        # Parameters
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))
        
        # Track reference positions at predicted theta values
        self.ref_positions_k = cvxpy.Parameter((2, self.config.TK + 1))  # [x_ref, y_ref]
        self.ref_positions_k.value = np.zeros((2, self.config.TK + 1))
        
        # Tangent and normal vectors at each prediction point
        self.sin_phi_k = cvxpy.Parameter(self.config.TK + 1)
        self.cos_phi_k = cvxpy.Parameter(self.config.TK + 1)
        self.sin_phi_k.value = np.zeros(self.config.TK + 1)
        self.cos_phi_k.value = np.zeros(self.config.TK + 1)
            
        objective = 0.0
        constraints = []
    

        for t in range(self.config.TK):
            objective += cvxpy.quad_form(self.uk[:, t], self.config.Rk)

        for t in range(self.config.TK - 1):
            u_diff = self.uk[:, t+1] - self.uk[:, t]
            objective += cvxpy.quad_form(u_diff, self.config.Rdk)
        
        # 3. MPCC-specific costs: Contouring error + Lag error + Progress maximization
        for t in range(self.config.TK + 1):
            # Position error
            dx = self.xk[0, t] - self.ref_positions_k[0, t]
            dy = self.xk[1, t] - self.ref_positions_k[1, t]
            
            # Contouring error: e_c = sin(phi) * dx + cos(phi) * dy
            e_c = self.sin_phi_k[t] * dx + self.cos_phi_k[t] * dy
            
            # Lag error: e_l = -cos(phi) * dx - sin(phi) * dy
            e_l = -self.cos_phi_k[t] * dx - self.sin_phi_k[t] * dy
            
            objective += self.q_contour * cvxpy.square(e_c)
            objective += self.q_lag * cvxpy.square(e_l)
            
        # 4. Progress maximization: Reward high v_theta values
        for t in range(self.config.TK):
            objective += self.q_theta * self.uk[2, t]  
        
        # ========== DYNAMICS CONSTRAINTS ==========
        # Vehicle dynamics for first (NXK-1) states
        A_block = []
        B_block = []
        C_block = []
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))
        
        for t in range(self.config.TK):
            # Get linearized dynamics for vehicle states (excluding theta for now)
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)
  
        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # Dynamics constraint: x[k+1] = A*x[k] + B*u[k] + C
        # This now includes: theta[k+1] = theta[k] + dt * v_theta[k]
        constraints += [
            cvxpy.vec(self.xk[:, 1:]) == 
            self.Ak_ @ cvxpy.vec(self.xk[:, :-1]) + 
            self.Bk_ @ cvxpy.vec(self.uk) + 
            self.Ck_
        ]
        
        # Initial condition
        constraints += [self.xk[:, 0] == self.x0k]

        # ========== STATE/INPUT CONSTRAINTS ==========

        state_constraints, input_constraints, input_diff_constraints = self.model.get_model_constraints()

        # ALL state constraints (including theta if your model provides them)
        for i in range(self.config.NXK):  
            constraints += [state_constraints[0, i] <= self.xk[i, :], 
                        self.xk[i, :] <= state_constraints[1, i]]

        # ALL input constraints (including v_theta if your model provides them)
        for i in range(self.config.NU):  
            constraints += [input_constraints[0, i] <= self.uk[i, :], 
                        self.uk[i, :] <= input_constraints[1, i]]
            constraints += [input_diff_constraints[0, i] <= cvxpy.diff(self.uk[i, :]),
                        cvxpy.diff(self.uk[i, :]) <= input_diff_constraints[1, i]]

        # Create optimization problem
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
                
    def mpc_prob_solve(self, x0):
        """
            Solve MPCC optimization problem
            
            x0: current state [x, y, v, phi, vx, vy, omega, theta] (NXK)
            theta_predictions: predicted theta values for updating references
        """  
        self.x0k.value = x0
        
        # Initialize path prediction
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        path_predict[:, 0] = x0
        
        if not np.any(np.isnan(self.states_output)):
            path_predict = self.states_output
        else:
            v_theta_nominal = 2.0  # Just a reasonable starting guess
            for t in range(1, self.config.TK + 1):
                path_predict[7, t] = path_predict[7, t-1] + self.config.DTK * v_theta_nominal
                path_predict[:7, t] = x0[:7]
        

        for t in range(self.config.TK + 1):
            theta_t = path_predict[7, t]
            x_ref, y_ref = self.TrackRef.get_position(theta_t)
            self.ref_positions_k.value[:, t] = [x_ref, y_ref]
            
            phi_ref = self.TrackRef.get_phi(theta_t)
            # Pre-compute sin and cos
            self.sin_phi_k.value[t] = np.sin(phi_ref)
            self.cos_phi_k.value[t] = np.cos(phi_ref)
            
        # 2. Update dynamics linearization
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))
        
        A_batch, B_batch, C_batch = self.model.batch_get_model_matrix(
            path_predict[:, :self.config.TK], 
            input_predict[:, :self.config.TK]
        )
        
        A_block = block_diag(tuple(A_batch))
        B_block = block_diag(tuple(B_batch))
        C_block = np.array(C_batch.flatten())
        
        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block
        
        # 3. Solve linearized problem
        self.MPC_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.01, 
                            eps_abs=0.001, eps_rel=0.001, verbose=False,
                            warm_start=True)   
        # Return final solution
        if self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE:
            ou = self.uk.value
            o_states = self.xk.value
        else:
            ou = np.zeros((self.config.NU, self.config.TK)) * np.NAN
            o_states = np.zeros((self.config.NXK, self.config.TK + 1)) * np.NAN
        
        return ou, o_states

    def MPCC_Control(self, x0):
        """
        Main MPCC control function
        
        x0: current vehicle state [x, y, v, phi, vx, vy, omega, theta] (NXK)
        """
        # Extract current theta from state (last element
        
        # Solve MPCC
        input_o, states_output = self.mpc_prob_solve(x0)
        
        if not np.any(np.isnan(states_output)):
            self.states_output = states_output
            self.input_o = input_o
        
        # Extract control output (3 inputs: accel, steering, v_theta)
        u = self.input_o[:, 0]
        
        # Generate reference path based on optimized theta trajectory
        ref_path_x = np.zeros(self.config.TK + 1)
        ref_path_y = np.zeros(self.config.TK + 1)
        for t in range(self.config.TK + 1):
            theta_t = states_output[self.theta_index, t] % self.track_length
            ref_path_x[t], ref_path_y[t] = self.TrackRef.get_position(theta_t)
        
        # Predicted trajectory from optimization
        pred_x = states_output[0, :]
        pred_y = states_output[1, :]

        return u, ref_path_x, ref_path_y, pred_x, pred_y, pred_x, pred_y
  
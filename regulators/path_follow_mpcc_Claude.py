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
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class TrackRef_MPCC:
    """
    Track reference system for MPCC.
    Provides:
    - Position (x, y) at any theta
    - Heading (phi) at any theta  
    - Tangent and normal vectors for contouring/lag error calculation
    """
    def __init__(self, waypoints):
        # Extract X and Y coordinates
        self.x = waypoints[:, 0]
        self.y = waypoints[:, 1]
        # Close the loop if needed
        dist_start_end = np.sqrt((self.x[0] - self.x[-1])**2 + (self.y[0] - self.y[-1])**2)
        
        if dist_start_end > 0.001:
            print(f"Closing the track loop... (Gap was {dist_start_end:.3f}m)")
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])
        else:
            self.x[-1] = self.x[0]
            self.y[-1] = self.y[0]
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dists = np.sqrt(dx**2 + dy**2)
        
        self.theta_arr = np.concatenate(([0], np.cumsum(dists)))
        self.track_length = self.theta_arr[-1]
        if self.track_length < 1e-6:
            raise ValueError(f"Track length too small: {self.track_length}m. Check waypoints.")
        
        # Create periodic splines
        self.spline_x = CubicSpline(self.theta_arr, self.x, bc_type='periodic')
        self.spline_y = CubicSpline(self.theta_arr, self.y, bc_type='periodic')

    def get_position(self, theta):
        """Returns x, y at given theta"""
        theta_wrapped = theta % self.track_length
        return float(self.spline_x(theta_wrapped)), float(self.spline_y(theta_wrapped))
    
    def get_phi(self, theta):
        """Returns the reference heading at given theta"""
        theta_wrapped = theta % self.track_length
        dx_d = self.spline_x(theta_wrapped, 1)
        dy_d = self.spline_y(theta_wrapped, 1)
        phi_ref = np.arctan2(dy_d, dx_d)
        return float(phi_ref)
    
    def get_tangent_vector(self, theta):
        """Returns unit tangent vector [t_x, t_y] at theta"""
        theta_wrapped = theta % self.track_length
        dx_d = self.spline_x(theta_wrapped, 1)
        dy_d = self.spline_y(theta_wrapped, 1)
        
        # Safe division - add small epsilon to prevent division by zero
        magnitude = np.sqrt(dx_d**2 + dy_d**2)
        magnitude_safe = np.maximum(magnitude, 1e-3)  # Ensure minimum magnitude
        
        return np.array([dx_d / magnitude_safe, dy_d / magnitude_safe])
    
    def get_normal_vector(self, theta):
        """Returns unit normal vector [n_x, n_y] at theta (perpendicular to tangent)"""
        tangent = self.get_tangent_vector(theta)
        # Rotate tangent by 90 degrees: (x, y) -> (-y, x)
        return np.array([-tangent[1], tangent[0]])
    
'''
class TrackRef_MPCC:
    """
    Track reference system for MPCC.

    Provides:
    - Position (x, y) at any theta
    - Heading (phi) at any theta
    - Tangent and normal vectors
    - Linearized contouring and lag error model
    """

    def __init__(self, waypoints):
        # Extract X and Y coordinates
        self.x = waypoints[:, 0]
        self.y = waypoints[:, 1]

        # Close the loop if needed
        dist_start_end = np.hypot(self.x[0] - self.x[-1],
                                  self.y[0] - self.y[-1])

        if dist_start_end > 1e-3:
            print(f"Closing the track loop... (Gap was {dist_start_end:.3f}m)")
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])
        else:
            self.x[-1] = self.x[0]
            self.y[-1] = self.y[0]

        # Arc-length parameterization
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dists = np.hypot(dx, dy)

        self.theta_arr = np.concatenate(([0.0], np.cumsum(dists)))
        self.track_length = self.theta_arr[-1]

        if self.track_length < 1e-6:
            raise ValueError("Track length too small")

        # Periodic splines
        self.spline_x = CubicSpline(self.theta_arr, self.x, bc_type="periodic")
        self.spline_y = CubicSpline(self.theta_arr, self.y, bc_type="periodic")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _wrap_theta(self, theta):
        return theta % self.track_length

    def get_position(self, theta):
        theta = self._wrap_theta(theta)
        return np.array([
            self.spline_x(theta),
            self.spline_y(theta)
        ])

    def get_phi(self, theta):
        theta = self._wrap_theta(theta)
        dx = self.spline_x(theta, 1)
        dy = self.spline_y(theta, 1)
        return float(np.arctan2(dy, dx))

    def get_tangent_vector(self, theta):
        theta = self._wrap_theta(theta)
        dx = self.spline_x(theta, 1)
        dy = self.spline_y(theta, 1)

        mag = np.hypot(dx, dy)
        mag = max(mag, 1e-3)

        return np.array([dx / mag, dy / mag])

    def get_normal_vector(self, theta):
        t = self.get_tangent_vector(theta)
        return np.array([-t[1], t[0]])

    # ------------------------------------------------------------------
    # Linearized MPCC errors
    # ------------------------------------------------------------------

    def linearized_contouring_lag_error(self, x, y, theta,
                                        x0, y0, theta0):
        """
        Linearized contouring and lag errors around (x0, y0, theta0)

        Args:
            x, y, theta : CVXPY variables or numpy scalars
            x0, y0, theta0 : linearization point (floats)

        Returns:
            e_c_lin, e_l_lin : linearized errors
        """

        # Reference at linearization point
        p_ref = self.get_position(theta0)
        t = self.get_tangent_vector(theta0)
        n = self.get_normal_vector(theta0)

        # Contouring error (no theta term)
        e_c = (
            n[0] * (x - p_ref[0])
            + n[1] * (y - p_ref[1])
        )

        # Lag error (includes theta term)
        e_l = (
            t[0] * (x - p_ref[0])
            + t[1] * (y - p_ref[1])
            - (theta - theta0)
        )

        return e_c, e_l
'''
            
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
        self.states_output = np.ones((self.config.NXK, self.config.TK)) * np.NaN
        
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
        """
        Generate reference trajectory based on predicted theta values.
        
        theta_array: predicted theta values for horizon [theta_0, theta_1, ..., theta_N]
        
        Returns reference trajectory with contouring/lag error coordinates
        """
        ref_traj = np.zeros((self.config.NXK + 2, self.config.TK + 1))
        
        for i, theta in enumerate(theta_array):
            theta_wrapped = theta % self.track_length
            
            # Reference position
            x_ref, y_ref = self.TrackRef.get_position(theta_wrapped)

            # For MPCC, we store reference in terms of contouring coordinates
            # The actual error will be calculated during optimization
            ref_traj[0, i] = x_ref
            ref_traj[1, i] = y_ref
      
        return ref_traj

    def mpc_prob_init(self):
        """
        Initialize MPCC optimization problem.
        State: [x, y, v, phi, vx, vy, omega, theta] (NXK)
        Control: [accel, steering, v_theta] (NU = 3)
        """
        # Decision variables
        self.xk = cvxpy.Variable((self.config.NXK, self.config.TK + 1))
        self.uk = cvxpy.Variable((self.config.NU, self.config.TK))  # Now includes v_theta
        
        # Parameters
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))
        
        # Track reference positions at predicted theta values
        self.ref_positions_k = cvxpy.Parameter((2, self.config.TK + 1))  # [x_ref, y_ref]
        self.ref_positions_k.value = np.zeros((2, self.config.TK + 1))
        
        # Tangent and normal vectors at each prediction point
        self.tangent_vectors_k = cvxpy.Parameter((2, self.config.TK + 1))
        self.normal_vectors_k = cvxpy.Parameter((2, self.config.TK + 1))
        self.tangent_vectors_k.value = np.zeros((2, self.config.TK + 1))
        self.normal_vectors_k.value = np.zeros((2, self.config.TK + 1))
        
        objective = 0.0
        constraints = []
        
        # ========== MPCC COST FUNCTION ==========
        
        # 1. Control effort cost (now 3 inputs: accel, steering, v_theta)

        for t in range(self.config.TK):
            objective += cvxpy.quad_form(self.uk[:, t], self.config.Rk)

        for t in range(self.config.TK - 1):
            u_diff = self.uk[:, t+1] - self.uk[:, t]
            objective += cvxpy.quad_form(u_diff, self.config.Rdk)
        
        # 3. MPCC-specific costs: Contouring error + Lag error + Progress maximization
        for t in range(self.config.TK + 1):
            # Position error vector (vehicle position - reference position)
            pos_error = cvxpy.vstack([
                self.xk[0, t] - self.ref_positions_k[0, t], 
                self.xk[1, t] - self.ref_positions_k[1, t]])
            
            # Contouring error: perpendicular distance to path (lateral deviation)
            # e_c = normal^T * position_error
            e_c = self.normal_vectors_k[0, t] * pos_error[0] +  self.normal_vectors_k[1, t] * pos_error[1]
            
            # Lag error: distance along path direction (longitudinal deviation)
            # e_l = tangent^T * position_error
            e_l = self.tangent_vectors_k[0, t] * pos_error[0] + self.tangent_vectors_k[1, t] * pos_error[1]
            
            objective += self.q_contour * cvxpy.square(e_c)
            objective += self.q_lag * cvxpy.square(e_l)
        
        # 4. Progress maximization: Reward high v_theta values
        # Note: q_theta is NEGATIVE to encourage progress
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
                
    def mpc_prob_solve(self, x0, theta_predictions):
        """
        Solve MPCC optimization problem
        
        x0: current state [x, y, v, phi, vx, vy, omega, theta] (NXK)
        theta_predictions: predicted theta values for updating references
        """
        self.x0k.value = x0
        
        # Update reference positions and vectors based on predicted theta
        for t in range(self.config.TK + 1):
            theta_t = theta_predictions[t] % self.track_length
            
            # Get reference position
            x_ref, y_ref = self.TrackRef.get_position(theta_t)
            self.ref_positions_k.value[:, t] = [x_ref, y_ref]
            
            # Get tangent and normal vectors
            tangent = self.TrackRef.get_tangent_vector(theta_t)
            normal = self.TrackRef.get_normal_vector(theta_t)
            # print("tangent:", tangent)
            # print("normal:", normal)
            self.tangent_vectors_k.value[:, t] = tangent
            self.normal_vectors_k.value[:, t] = normal
        
        # Update vehicle dynamics matrices
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))
        
        # Use previous solution if available for warm start
        if not np.any(np.isnan(self.states_output)):
            path_predict = self.states_output
            
        A_batch, B_batch, C_batch = self.model.batch_get_model_matrix(
            path_predict[:, :self.config.TK], 
            input_predict[:, :self.config.TK]
        )
        # print(A_batch[0])
        # Create block diagonal matrices
        A_block = block_diag(tuple(A_batch))
        B_block = block_diag(tuple(B_batch))
        C_block = np.array(C_batch.flatten())
        
        # Update parameters
        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block
        
        # Solve
        self.MPC_prob.solve(
            solver=cvxpy.OSQP, 
            polish=True, 
            adaptive_rho=True, 
            rho=0.01,
            eps_abs=0.0005, 
            eps_rel=0.0005, 
            verbose=False,
            warm_start=True
        )
        
        if self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE:
            ou = self.uk.value
            o_states = self.xk.value
        else:
            print("Error: Cannot solve MPCC... Status:", self.MPC_prob.status)
            ou = np.zeros((self.config.NU, self.config.TK)) * np.NAN
            o_states = np.zeros((self.config.NXK, self.config.TK + 1)) * np.NAN
            
        return ou, o_states

    def MPCC_Control(self, x0):
        """
        Main MPCC control function
        
        x0: current vehicle state [x, y, v, phi, vx, vy, omega, theta] (NXK)
        """
        # Extract current theta from state (last element)
        theta_current = x0[self.theta_index]
        
        # Predict theta progression for reference generation
        if not np.any(np.isnan(self.states_output)):
            # Use previous solution's theta trajectory
            theta_predictions = self.states_output[self.theta_index, :]
            
            # Ensure we have TK+1 predictions
            if theta_predictions.shape[0] < self.config.TK + 1:
                # Extrapolate if needed
                v_theta_last = self.input_o[2, -1] if not np.isnan(self.input_o[2, -1]) else 5.0
                extra = self.config.TK + 1 - theta_predictions.shape[0]
                theta_pred_extend = np.arange(1, extra + 1) * v_theta_last * self.config.DTK
                theta_predictions = np.concatenate([theta_predictions, 
                                                   theta_predictions[-1] + theta_pred_extend])
        else: 
            v_theta_init = 0.0 
            theta_predictions = theta_current + np.arange(self.config.TK + 1) * v_theta_init * self.config.DTK
        
        # Solve MPCC
        input_o, states_output = self.mpc_prob_solve(x0, theta_predictions)
        
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
import time
from dataclasses import dataclass, field
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
import copy
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

class TrackRef_MPCC:
    def __init__(self, waypoints):
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
        
    def get_current_theta(self, vehicle_state, theta_prev):
        x_car, y_car = vehicle_state[0], vehicle_state[1]
        def cost_fn(t):
            t_wrapped = t % self.track_length
            xt = self.spline_x(t_wrapped)
            yt = self.spline_y(t_wrapped)
            return (x_car - xt)**2 + (y_car - yt)**2

        search_min = theta_prev

        search_max = theta_prev + 5.0

        res = minimize_scalar(
            cost_fn,
            bounds=(search_min, search_max),
            method='bounded',
            options={'xatol': 1e-4})

        theta_k = res.x
        return theta_k % self.track_length

class STMPCCPlannerCasadi:
    def __init__(self, model, config, waypoints=None, track=None):
        self.waypoints = waypoints
        self.model = model
        self.config = config
        self.track = track
        
        self.TrackRef = TrackRef_MPCC(self.waypoints)
        self.track_length = self.TrackRef.track_length
        self.theta_index = self.config.NXK - 1
        
        self.input_o = np.zeros(self.config.NU) * np.NAN  # NU = 2 (accel, steering)
        self.states_output = np.ones((self.config.NXK, self.config.TK + 1)) * np.NaN
        
        self.q_contour = self.config.q_contour      # Contouring error weight
        self.q_lag = self.config.q_lag              # Lag error weight  
        self.q_theta = self.config.q_theta          # Progress maximization (negative = reward)
        
        self.DTK = self.config.DTK
        self.MASS = self.config.MASS
        self.I_Z = self.config.I_Z
        self.LF = self.config.LF
        self.LR = self.config.LR
        self.TORQUE_SPLIT = self.config.TORQUE_SPLIT
        self.BR = self.config.BR
        self.DR = self.config.DR
        self.BF = self.config.BF
        self.CF = self.config.CF
        self.DF = self.config.DF
        self.CM = self.config.CM
        self.CR = self.config.CR
        self.CR0 = self.config.CR0
        self.CR2 = self.config.CR2
        self.theta_prev = 0.0
        self.mpc_prob_init()

    def plan(self, states, waypoints=None):
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
    
    def predictive_model(self, state, control_input):
        # OPTION B: state = [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements, NO theta)
        # control_input = [Fxr, delta_v]
        # Theta evolution is handled separately

        x = state[0]
        y = state[1]
        vx = state[2]
        yaw = state[3]
        vy = state[4]
        yaw_rate = state[5]
        steering_angle = state[6]
        
        Fxr = control_input[0]
        delta_v = control_input[1]

        # Safe velocity handling
        vx_safe = ca.fmax(ca.fabs(vx), 0.05)
        vx_safe = ca.sign(vx) * vx_safe

        # Tire slip angles
        alfa_f = steering_angle - ca.atan2(yaw_rate * self.LF + vy, vx_safe)
        alfa_r = ca.atan2(yaw_rate * self.LR - vy, vx_safe)

        # Pacejka tire model
        Ffy = self.DF * ca.sin(self.CF * ca.atan(self.BF * alfa_f))
        Fry = self.DR * ca.sin(self.CR * ca.atan(self.BR * alfa_r))

        # Longitudinal forces
        Fx = self.CM * Fxr - self.CR0 - self.CR2 * vx_safe ** 2.0
        Frx = Fx * (1.0 - self.TORQUE_SPLIT)
        Ffx = Fx * self.TORQUE_SPLIT

        # Vehicle dynamics (7 states)
        dx = vx_safe * ca.cos(yaw) - vy * ca.sin(yaw)
        dy = vx_safe * ca.sin(yaw) + vy * ca.cos(yaw)
        dvx = (1.0 / self.MASS) * (Frx - Ffy * ca.sin(steering_angle) + Ffx * ca.cos(steering_angle) + vy * yaw_rate * self.MASS)
        dyaw = yaw_rate
        dvy = (1.0 / self.MASS) * (Fry + Ffy * ca.cos(steering_angle) + Ffx * ca.sin(steering_angle) - vx_safe * yaw_rate * self.MASS)
        dyaw_rate = (1.0 / self.I_Z) * (Ffy * self.LF * ca.cos(steering_angle) - Fry * self.LR)
        dsteering = delta_v
        
        f = ca.vertcat(dx, dy, dvx, dyaw, dvy, dyaw_rate, dsteering)
        
        return f
    def rk4_step(self, x, u):
        """
        Standard RK4 integration step.
        x: state at time t
        u: control input
        """
        dt = self.DTK
        
        # Calculate the four slopes
        k1 = self.predictive_model(x, u)
        k2 = self.predictive_model(x + dt/2 * k1, u)
        k3 = self.predictive_model(x + dt/2 * k2, u)
        k4 = self.predictive_model(x + dt * k3, u)
        
        # Weighted average update
        x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def mpc_prob_init(self):
        # OPTION B: Separate decision variables for states and theta
        self.xk = ca.MX.sym('xk', self.config.NXK, self.config.TK + 1)  # NXK = 7 (no theta)
        self.uk = ca.MX.sym('uk', self.config.NU, self.config.TK)        # NU = 2
        self.theta_k = ca.MX.sym('theta_k', self.config.TK + 1)          # Theta as separate variable
        self.vik = ca.MX.sym('vik', self.config.TK)                      # v_theta
        
        # Parameters
        self.x0k = ca.MX.sym('x0k', self.config.NXK)
        self.theta0 = ca.MX.sym('theta0')  # Initial theta
        
        # Track reference will be computed from theta_k symbolically
        # We'll use lookup tables passed as parameters
        self.ref_x_lut = ca.MX.sym('ref_x_lut', self.config.TK + 1)
        self.ref_y_lut = ca.MX.sym('ref_y_lut', self.config.TK + 1)
        self.sin_phi_lut = ca.MX.sym('sin_phi_lut', self.config.TK + 1)
        self.cos_phi_lut = ca.MX.sym('cos_phi_lut', self.config.TK + 1)

        objective = 0.0
        constraints = []
        lbg = []
        ubg = []

        # Dynamics constraints for vehicle states
        for t in range(self.config.TK):
            f = self.predictive_model(self.xk[:, t], self.uk[:, t])

            
            x_next = self.rk4_step(self.xk[:, t], self.uk[:, t])
            
            constraints.append(self.xk[:, t + 1] - x_next)
            lbg.extend([0.0] * self.config.NXK)
            ubg.extend([0.0] * self.config.NXK)

        # Theta dynamics constraints
        for t in range(self.config.TK):
            theta_next = self.theta_k[t] + self.DTK * self.vik[t]
            constraints.append(self.theta_k[t + 1] - theta_next)
            lbg.append(0.0)
            ubg.append(0.0)

        # Cost function - contouring and lag errors
        for t in range(self.config.TK + 1):
            # Position error relative to reference at theta_k[t]
            dx = self.xk[0, t] - self.ref_x_lut[t]
            dy = self.xk[1, t] - self.ref_y_lut[t]
            
            # Contouring error (perpendicular to path)
            e_c = -self.sin_phi_lut[t] * dx + self.cos_phi_lut[t] * dy
            
            # Lag error (along path)
            e_l = self.cos_phi_lut[t] * dx + self.sin_phi_lut[t] * dy
            
            objective += self.q_contour * e_c ** 2
            objective += self.q_lag * e_l ** 2

        # Progress reward - maximize theta progression (negative cost)
        for t in range(self.config.TK):
            objective += -self.q_theta * self.vik[t]


        # Input control effort
        for t in range(self.config.TK):
            p_u_1 = self.uk[0, t]
            p_u_2 = self.uk[1, t]
            p_vi = self.vik[t]
            p_u = ca.vertcat(p_u_1, p_u_2, p_vi)
            objective += p_u.T@self.config.Rk_ca@ p_u

        # Input smoothness
        for t in range(self.config.TK - 1):
            du_1 = self.uk[0, t + 1] - self.uk[0, t]
            du_2 = self.uk[1, t + 1] - self.uk[1, t]
            dvi = self.vik[t + 1] - self.vik[t]
            du = ca.vertcat(du_1, du_2, dvi)
            objective += du.T@self.config.Rdk_ca@ du

        # Initial condition constraints
        constraints.append(self.xk[:, 0] - self.x0k)
        lbg.extend([0.0] * self.config.NXK)
        ubg.extend([0.0] * self.config.NXK)
        
        # Initial theta constraint
        constraints.append(self.theta_k[0] - self.theta0)
        lbg.append(0.0)
        ubg.append(0.0)

        # Concatenate constraints
        g = ca.vertcat(*constraints)

        # Decision variable vector
        opt_variables = ca.vertcat(
            ca.reshape(self.xk, -1, 1),      # States (7 x (TK+1))
            ca.reshape(self.uk, -1, 1),      # Controls (2 x TK)
            self.theta_k,                     # Theta (TK+1)
            self.vik                          # v_theta (TK)
        )

        # Parameter vector
        opt_params = ca.vertcat(
            self.x0k,
            self.theta0,
            self.ref_x_lut,
            self.ref_y_lut,
            self.sin_phi_lut,
            self.cos_phi_lut
        )

        # Create NLP
        nlp = {
            'x': opt_variables,
            'f': objective,
            'g': g,
            'p': opt_params
        }

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-5,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes'
        }

 

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store sizes for unpacking
        self.n_states = self.config.NXK * (self.config.TK + 1)
        self.n_controls = self.config.NU * self.config.TK
        self.n_theta = self.config.TK + 1
        self.n_vi = self.config.TK

        # Bounds on decision variables
        self.lbx = []
        self.ubx = []

        # State bounds (7 states: x, y, vx, yaw, vy, yaw_rate, steering)
        for t in range(self.config.TK + 1):
            self.lbx.extend([
                -np.inf,                    # x
                -np.inf,                    # y
                self.config.MIN_SPEED,      # vx
                -np.inf,                    # yaw
                -np.inf,                    # vy
                -np.inf,                    # yaw_rate
                self.config.MIN_STEER       # steering_angle
            ])
            self.ubx.extend([
                np.inf,                     # x
                np.inf,                     # y
                self.config.MAX_SPEED,      # vx
                np.inf,                     # yaw
                np.inf,                     # vy
                np.inf,                     # yaw_rate
                self.config.MAX_STEER       # steering_angle
            ])

        # Control bounds
        for t in range(self.config.TK):
            self.lbx.extend([
                self.config.MAX_DECEL * self.MASS,  # Fxr
                -self.config.MAX_STEER_V             # delta_v
            ])
            self.ubx.extend([
                self.config.MAX_ACCEL * self.MASS,  # Fxr
                self.config.MAX_STEER_V              # delta_v
            ])

        # Theta bounds
        min_theta = getattr(self.config, 'MIN_THETA', 0.0)
        max_theta = getattr(self.config, 'MAX_THETA', self.track_length * 10)
        for t in range(self.config.TK + 1):
            self.lbx.append(min_theta)
            self.ubx.append(max_theta)

        # v_theta bounds
        min_vi = getattr(self.config, 'MIN_VI', 0.0)
        max_vi = getattr(self.config, 'MAX_VI', 10.0)
        for t in range(self.config.TK):
            self.lbx.append(min_vi)
            self.ubx.append(max_vi)

        self.lbg = lbg
        self.ubg = ubg
        
        # Initial guess
        self.x0_opt = np.zeros(self.n_states + self.n_controls + self.n_theta + self.n_vi)

    def mpc_prob_solve(self, x0):
        """
        x0 should be [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements)
        theta is handled separately
        """
        # Get current theta from vehicle position
        theta_0 = self.TrackRef.get_current_theta(x0, self.theta_prev)
        self.theta_prev = theta_0
        
        # Estimate v_theta based on current velocity
        v_theta_est = 2.0
        
        # Generate predicted theta trajectory
        theta_pred = np.zeros(self.config.TK + 1)
        theta_pred[0] = theta_0

        for t in range(1, self.config.TK + 1):
            theta_pred[t] = theta_pred[t - 1] + self.DTK * v_theta_est

        # Compute reference trajectory and track angles from predicted theta
        ref_x = np.zeros(self.config.TK + 1)
        ref_y = np.zeros(self.config.TK + 1)
        sin_phi = np.zeros(self.config.TK + 1)
        cos_phi = np.zeros(self.config.TK + 1)

        for t in range(self.config.TK + 1):
            theta_t = theta_pred[t] % self.track_length
            ref_x[t], ref_y[t] = self.TrackRef.get_position(theta_t)
            phi_t = self.TrackRef.get_phi(theta_t)
            sin_phi[t] = np.sin(phi_t)
            cos_phi[t] = np.cos(phi_t)

        # Build parameter vector
        p = np.concatenate([
            x0,
            [theta_0],
            ref_x,
            ref_y,
            sin_phi,
            cos_phi
        ])
        
        # Warm start with previous solution or initial guess
        if not np.any(np.isnan(self.states_output)):
            # Use previous solution as warm start
            x0_states = self.states_output.T.flatten()
            x0_controls = self.input_o.T.flatten()
            x0_theta = self.theta_output
            x0_vi = np.ones(self.n_vi) * v_theta_est
            self.x0_opt = np.concatenate([x0_states, x0_controls, x0_theta, x0_vi])
        else:
            # Initial guess
            x0_states = np.tile(x0, self.config.TK + 1)
            x0_controls = np.zeros(self.n_controls)
            x0_theta = theta_pred
            x0_vi = np.ones(self.n_vi) * v_theta_est
            self.x0_opt = np.concatenate([x0_states, x0_controls, x0_theta, x0_vi])
        
        # Solve the optimization problem
        try:
            sol = self.solver(
                x0=self.x0_opt,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # Extract solution
            x_opt = sol['x'].full().flatten()
            
            # Unpack states, controls, theta, and v_theta
            idx = 0
            states = x_opt[idx:idx + self.n_states].reshape((self.config.TK + 1, self.config.NXK)).T
            idx += self.n_states
            
            controls = x_opt[idx:idx + self.n_controls].reshape((self.config.TK, self.config.NU)).T
            idx += self.n_controls
            
            theta = x_opt[idx:idx + self.n_theta]
            idx += self.n_theta
            
            vi = x_opt[idx:idx + self.n_vi]
            
            # Update warm start for next iteration
            self.x0_opt = x_opt
            
            return controls, states, theta
                
        except Exception as e:
            print(f"Optimization error: {e}")
            controls = np.zeros((self.config.NU, self.config.TK))
            states = np.tile(x0[:, None], (1, self.config.TK + 1))
            theta = theta_pred
            return controls, states, theta

    def MPCC_Control(self, x0_full):
        """
        x0_full can be either:
        - [x, y, vx, yaw, vy, yaw_rate, steering_angle] (7 elements)
        - [x, y, vx, yaw, vy, yaw_rate, steering_angle, theta] (8 elements, theta ignored)
        """
        # Extract only the first 7 states (ignore theta if present)
        x0 = x0_full[:self.config.NXK]
        
        # Solve MPCC
        input_o, states_output, theta_output = self.mpc_prob_solve(x0)
        
        if not np.any(np.isnan(states_output)):
            self.states_output = states_output
            self.input_o = input_o
            self.theta_output = theta_output
        
        # Extract control output (2 inputs: Fxr, delta_v)
        u = self.input_o[:, 0]
        
        # Generate reference path based on optimized theta trajectory
        ref_path_x = np.zeros(self.config.TK + 1)
        ref_path_y = np.zeros(self.config.TK + 1)

        for t in range(self.config.TK + 1):
            theta_t = self.theta_output[t] % self.track_length
            ref_path_x[t], ref_path_y[t] = self.TrackRef.get_position(theta_t)
        
        # Predicted trajectory from optimization
        pred_x = states_output[0, :]
        pred_y = states_output[1, :]

        return u, ref_path_x, ref_path_y, pred_x, pred_y, pred_x, pred_y
  
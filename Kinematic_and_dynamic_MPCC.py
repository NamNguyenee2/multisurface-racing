import time
import yaml
import gym
from argparse import Namespace
from regulators.pure_pursuit import *
from regulators.path_follow_mpcc import STMPCPlanner
from regulators.path_follow_mpcc_Claude import *
from models.extended_kinematic_mpcc import ExtendedKinematicModel
from models.dynamic_mpcc import DynamicBicycleModel
from helpers.closest_point import *
import numpy as np

from pyglet.gl import GL_POINTS
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

'''
@dataclass
class MPCConfigEXT:
    NXK: int = 8  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 3  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 20  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000000001, 10.0, 1.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000000001, 10.0, 1.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 10.5, 15.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 10.5, 15.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MAX_THETA: float = 1000.0  # maximum heading angle [deg]
    MIN_THETA: float = 0.0  # minimum heading angle [deg]
    MAX_VI: float = 45.0 
    MIN_VI: float = 0.0
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    MASS: float = 1225.887  # Vehicle mass
'''

@dataclass
class MPCConfigDYN:
    NXK: int = 8  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 3  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 10  # finite time horizon length kinematic

    Rk: list = field(
        default_factory=lambda: np.diag([0.000000001, 1.0, 1.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.000000001, 1.0, 1.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        default_factory=lambda: np.diag([20.5, 20.5, 0., 0.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # state error cost matrix, for the next (T) prediction time steps
    Qfk: list = field(
        default_factory=lambda: np.diag([20.5, 20.5, 0., 0.0, 0.0, 0.0, 0.0, 0.0])
        # [13.5, 13.5, 5.5, 13.0, 0.0, 0.0, 0.0]
    )  # final state error matrix, penalty  for the final state constraints
    q_contour: float = 30.0
    q_lag: float = 100.0
    q_theta: float = -10000.0
    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.05  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MAX_THETA: float = 1000.0  # maximum heading angle [deg]
    MIN_THETA: float = 0.0  # minimum heading angle [deg]
    MAX_VI: float = 45.0 
    MIN_VI: float = 0.0
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    # model parameters
    MASS: float = 1225.887  # Vehicle mass
    I_Z: float = 1560.3729  # Vehicle inertia
    TORQUE_SPLIT: float = 0.0  # Torque distribution

    BR: float = 15.9504  # Pacejka tire model parameter B - rear tire
    CR: float = 1.3754  # Pacejka tire model parameter C - rear tire
    DR: float = 4500.9280  # Pacejka tire model parameter D - rear tire

    BF: float = 9.4246  # Pacejka tire model parameter B - front tire
    CF: float = 5.9139  # Pacejka tire model parameter C - front tire
    DF: float = 4500.8218  # Pacejka tire model parameter D - front tire

    # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
    CM: float = 0.9459
    CR0: float = 2.3451
    CR2: float = -0.0095


def draw_point(e, point, colour):
    scaled_point = 50. * point
    ret = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_point[0], scaled_point[1], 0]), ('c3B/stream', colour))
    return ret


class DrawDebug:
    def __init__(self):
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.f = 0

    def draw_debug(self, e):
        # delete dynamic objects
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        # spawn new objects
        for p in self.reference_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0]))

        for p in self.predicted_traj_show:
            self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0]))

class TrackRef:
    def __init__(self, waypoints):
        """
        Step 1: Create continuous track parameterization.
        waypoints: np.array shape (N, M), columns [x, y, ...]
        """
        # Extract X and Y coordinates
        self.x = waypoints[:, 0]
        self.y = waypoints[:, 1]
        
        # --- FIX: FORCE CLOSE THE LOOP ---
        # Check if the last point is far from the first point
        dist_start_end = np.sqrt((self.x[0] - self.x[-1])**2 + (self.y[0] - self.y[-1])**2)
        
        if dist_start_end > 0.001: # If gap is larger than 1mm
            print(f"Closing the track loop... (Gap was {dist_start_end:.3f}m)")
            # Append the starting point to the end
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])
        else:
            # Even if they are close, enforce exact equality for scipy
            self.x[-1] = self.x[0]
            self.y[-1] = self.y[0]

        # 1. Calculate distance between each consecutive point
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        dists = np.sqrt(dx**2 + dy**2)
        
        # 2. Cumulative sum to get theta (arc length) for each point
        # [0, d1, d1+d2, ..., Length]
        self.theta_arr = np.concatenate(([0], np.cumsum(dists)))
        self.track_length = self.theta_arr[-1]
        
        # 3. Create Splines: X(theta) and Y(theta)
        # bc_type='periodic' ensures the finish line connects smoothly to start
        self.spline_x = CubicSpline(self.theta_arr, self.x, bc_type='periodic')
        self.spline_y = CubicSpline(self.theta_arr, self.y, bc_type='periodic')
        
    def get_current_theta(self, vehicle_state):
        """
        Calculates the exact theta (arc length) closest to the vehicle.
        Combines Coarse Search (discrete) + Fine Search (optimization).
        """
        x_car = vehicle_state[0]
        y_car = vehicle_state[1]

        # --- A. Coarse Search (Discrete) ---
        # Check all discrete waypoints to find the approximate closest one.
        dx = self.x - x_car
        dy = self.y - y_car
        dists_sq = dx**2 + dy**2
        
        min_index = np.argmin(dists_sq)
        theta_guess = self.theta_arr[min_index]

        # --- B. Fine Search (Continuous Optimization) ---
        # Define cost function: Squared distance from car to spline at theta
        def cost_fn(theta):
            # Wrap theta for circular tracks (e.g., -1 becomes 99)
            theta_wrapped = theta % self.track_length
            x_tr = self.spline_x(theta_wrapped)
            y_tr = self.spline_y(theta_wrapped)
            # print(self.track_length)
            return (x_car - x_tr)**2 + (y_car - y_tr)**2
        # We allow bounds to go negative or > length because cost_fn handles wrapping
        window = 2.0 
        bnds = (theta_guess - window, theta_guess + window)
        res = minimize_scalar(cost_fn, bounds=bnds, method='bounded')
        # Final result wrapped to [0, track_length]
        theta_k = res.x % self.track_length
        return theta_k

def main():  # after launching this you can run visualization.py to see the results
    """
    main entry point
    """

    # Choose program parameters
    model_to_use = 'dynamic'  # options: ext_kinematic, pure_pursuit, dynamic
    map_name = 'rounded_rectangle'  # Nuerburgring, SaoPaulo, rounded_rectangle, l_shape, BrandsHatch, DualLaneChange, Austin, Budapest, Catalunya
    # Hockenheim, IMS, Melbourne, MexicoCity, Montreal, Monza, MoscowRaceway, Oschersleben, Sakhir, Sepang, Silverstone, Sochi, Spa, Spielberg
    # YasMarina
    rotate_map = False  # !!!! If the car is spawning with bad orientation change value here !!!! TODO Fix here so this is not needed anymore
    use_dyn_friction = False
    constant_friction = 0.5
    control_step = 100.0  # ms
    render_every = 40  # render graphics every n simulation steps
    constant_speed = False
    constant_speed_value = 8.0
    velocity_profile_multiplier = 0.9
    number_of_laps = 50
    start_point = 1  # index on the trajectory to start from

    # ekin_config = MPCConfigEXT()
    dyn_config = MPCConfigDYN()
    # ekin_config.DTK = kin_config.DTK = dyn_config.DTK = control_step / 1000.0

    # Creating the single-track Motion planner and Controller

    # Init Pure-Pursuit regulator
    work = {'mass': 1225.88, 'lf': 0.80597534362552312, 'tlad': 10.6461887897713965, 'vgain': 1.0}

    # Load map config file
    with open('configs/config_%s.yaml' % map_name) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if use_dyn_friction:
        if map_name == 'l_shape':
            tpamap_name = './maps/l_shape/friction_data/l_shape_l720_track_tpamap.csv'
            tpadata_name = './maps/l_shape/friction_data/l_shape_l720_track_tpadata.json'
        if map_name == 'DualLaneChange':
            # tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpamap.csv'
            # tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange5z_track_tpamap.csv'
            tpamap_name = './maps/DualLaneChange/friction_data/DualLaneChange3zv2_track_tpamap.csv'
            # tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange_track_tpadata.json'
            # tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange5z_track_tpadata.json'
            tpadata_name = './maps/DualLaneChange/friction_data/DualLaneChange3zv2_track_tpadata.json'
        if map_name == 'SaoPaulo':
            tpamap_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpamap.csv'
            tpadata_name = './maps/SaoPaulo/friction_data/SaoPaulo_track_tpadata.json'
        if map_name == 'Nuerburgring':
            tpamap_name = './maps/Nuerburgring/friction_data/Nuerburgring_tpamap.csv'
            tpadata_name = './maps/Nuerburgring/friction_data/Nuerburgring_tpadata.json'

        tpamap = np.loadtxt(tpamap_name, delimiter=';', skiprows=1)

        tpadata = {}
        with open(tpadata_name) as f:
            tpadata = json.load(f)

    raceline = np.loadtxt(conf.wpt_path, delimiter=";", skiprows=3)
    waypoints = np.array(raceline)
    track = TrackRef(waypoints)
    if rotate_map == True:
        waypoints[:, 3] += 1.5707963268

    if constant_speed:
        # waypoints[:, 5] = waypoints[:, 5] * 2.0
        waypoints[:, 5] = np.ones((waypoints[:, 5].shape[0],)) * constant_speed_value
    else:
        waypoints[:, 5] *= velocity_profile_multiplier

    # init controllers
    planner_pp = PurePursuitPlanner(conf, 0.805975 + 1.50876)  # 0.805975 + 1.50876
    planner_pp.waypoints = waypoints

    # planner_ekin_mpc = STMPCPlanner(model=ExtendedKinematicModel(config=MPCConfigEXT()), waypoints=waypoints,
    #                                 config=MPCConfigEXT())

    planner_dyn_mpc = STMPCPlanner(model=DynamicBicycleModel(config=dyn_config), waypoints=waypoints,
                                   config=dyn_config)

    # print("waypoints shape:", waypoints[0])
    # init graphics
    draw = DrawDebug()

    def render_callback(env_renderer):

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 7000
        e.left = left - 3000
        e.right = right + 3000
        e.top = top + 3000
        e.bottom = bottom - 3000

        planner_pp.render_waypoints(e)
        draw.draw_debug(e)

    # MB - reference point: center of mass
    # dynamic_ST - reference point: center of mass

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=0.001, model='MB', drive_control_mode='acc',
                   steering_control_mode='vel')

    env.add_render_callback(render_callback)
    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        np.array([[waypoints[start_point, 1], waypoints[start_point, 2], waypoints[start_point, 3], 0.0, 8.0, 0.0, 0.0]]))
    env.render()

    laptime = 0.0
    start = time.time()
    last_render = 0

    # init logger
    log = {'time': [], 'x': [], 'y': [], 'lap_n': [], 'vx': [], 'v_ref': [], 'tracking_error': []}

    # calc number of sim steps per one control step
    num_of_sim_steps = int(control_step / (env.timestep * 1000.0))


    # print('Model used: %s' % model_to_use)
    
    while not done:
        
        # print("env.sim.agents[0].state:", env.sim.agents[0].state)
        # Regulator step MPC
        vehicle_state = np.array([env.sim.agents[0].state[0],  # x
                                  env.sim.agents[0].state[1],  # y
                                  env.sim.agents[0].state[3],  # vx
                                  env.sim.agents[0].state[4],  # yaw angle
                                  env.sim.agents[0].state[10],  # vy
                                  env.sim.agents[0].state[5],  # yaw rate
                                  env.sim.agents[0].state[2],  # steering angle
                                  ]) + np.random.randn(7) * 0.00001
        theta_k = track.get_current_theta(vehicle_state)
        vehicle_state = np.concatenate((vehicle_state, [theta_k]))

        # print("theta_k:", theta_k, "x_ref:", x_ref, "y_ref:", y_ref, "phi_ref:", phi_k)
        
        # augmented_state = [vehicle_state.copy(), theta_MPCC]
        # break
        u = [1.0, 0.0, 0.0]
        '''
        if model_to_use == 'pure_pursuit':
            # Regulator step pure pursuit
            speed, steer_angle = planner_pp.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                                 work['tlad'],
                                                 work['vgain'])

            draw.reference_traj_show = np.array([[obs['poses_x'][0]], [obs['poses_y'][0]]]).T

            error_steer = steer_angle - env.sim.agents[0].state[2]
            u[1] = 10.0 * error_steer

            error_drive = speed - env.sim.agents[0].state[3]
            u[0] = 8.0 * error_drive

        elif model_to_use == "ext_kinematic":
            u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_ekin_mpc.plan(
                vehicle_state)
            u[0] = u[0] / planner_ekin_mpc.config.MASS  # Force to acceleration
            # u[0] = -u[1]
            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T
            print("use ext_kinematic")

        '''
        if model_to_use == "dynamic":
            if vehicle_state[2] < 0.1:
                u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_dyn_mpc.plan(
                    vehicle_state)
            else:
                u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy = planner_dyn_mpc.plan(
                    vehicle_state)

            u[0] = u[0] / planner_dyn_mpc.config.MASS  # Force to acceleration

            # draw predicted states and reference trajectory
            draw.reference_traj_show = np.array([mpc_ref_path_x, mpc_ref_path_y]).T
            draw.predicted_traj_show = np.array([mpc_pred_x, mpc_pred_y]).T

        _, tracking_error, _, _, n_point = nearest_point_on_trajectory(np.array([env.sim.agents[0].state[0], env.sim.agents[0].state[1]]),
                                                                       np.array([waypoints[:, 1], waypoints[:, 2]]).T)

        # set correct friction to the environment
        if use_dyn_friction:
            min_id = get_closest_point_vectorized(np.array([obs['poses_x'][0], obs['poses_y'][0]]), np.array(tpamap))
            env.params['tire_p_dy1'] = tpadata[str(min_id)][0] * 0.9  # mu_y
            env.params['tire_p_dx1'] = tpadata[str(min_id)][0]  # mu_x
        else:
            env.params['tire_p_dy1'] = constant_friction #* 0.9  # mu_y
            env.params['tire_p_dx1'] = constant_friction  # mu_x

   

        print('Model used: %s' % model_to_use)
        # Simulation step
        step_reward = 0.0
        for i in range(num_of_sim_steps):
            obs, rew, done, info = env.step(np.array([[u[1], u[0]]]))
            step_reward += rew

            # Rendering
            last_render += 1
            if last_render >= render_every:
                last_render = 0
                env.render(mode='human_fast')

        laptime += step_reward

        # Logging
        log['time'].append(laptime)
        log['x'].append(env.sim.agents[0].state[0])
        log['y'].append(env.sim.agents[0].state[1])
        log['vx'].append(env.sim.agents[0].state[3])
        log['v_ref'].append(waypoints[:, 5][n_point])
        log['tracking_error'].append(tracking_error)
        log['lap_n'].append(obs['lap_counts'][0])



        if obs['lap_counts'][0] == number_of_laps:  # or env.sim.agents[0].state[3] < 0.4 or tracking_error > 10.0:
            done = 1

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
    with open('log01_eval', 'w') as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()


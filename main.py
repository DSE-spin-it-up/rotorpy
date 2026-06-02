import numpy as np
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.dronado_params import quad_params
from rotorpy.controllers.tricopter_control import SE3Control
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import SinusoidWind

sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified.
                           controller=SE3Control(quad_params),        # controller object, must be specified.
                           trajectory=TwoDLissajous(),                # trajectory object, must be specified.
                           wind_profile=SinusoidWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind.
                           sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                           imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                           mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.
                           estimator    = None,                       # OPTIONAL: estimator object
                           world        = None,                      # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                           safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                       )

x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
sim_instance.vehicle.initial_state = x0

# The results are a dictionary containing the relevant state, input, and measurements vs time.
results = sim_instance.run(t_final      = 20,       # The maximum duration of the environment in seconds
                           use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates.
                           terminate    = False,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                           plot            = True,     # Boolean: plots the vehicle states and commands
                           plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                           plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                           plot_imu        = True,     # Boolean: plots the IMU measurements
                           animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play.
                           animate_wind    = True,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV.
                           verbose         = True,     # Boolean: will print statistics regarding the simulation.
                           fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/.
                    )
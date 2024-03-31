# All libraries are imported in the children classes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from numpy import sin, cos
from scipy.integrate import odeint, solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, HTMLWriter
from time import perf_counter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 12


class Simulation:

    REQUIRED_SETUP = ["t_sim"]
    DEFAULT_SETUP = {"oversample": 1, "slowdown": 1., "fps": 30}

    def __init__(self, setup, params, initials, REQUIRED_PARAMS, REQUIRED_INITIALS):
        self.validate_setup(setup)
        self.validate_input(params, REQUIRED_PARAMS, "params")
        self.validate_input(initials, REQUIRED_INITIALS, "initials")

        # 'slowdown' : . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        # 'oversample :  display one frame every ... framess
        for k in setup:
            setattr(self, k, setup[k])
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        self.params = params
        for k in params:
            setattr(self, k, params[k])
        self.initials = initials
        for k in initials:
            setattr(self, k, initials[k])

        self.full_t = np.linspace(0., self.t_sim, self.n_steps+1)
        self.t = self.full_t[::self.oversample]
    
        return
    
    def validate_setup(self, setup):
        missing_keys = [key for key in self.REQUIRED_SETUP if key not in setup]
        if missing_keys:
            raise ValueError(f"Missing required setup keys: {missing_keys}")
        
        for key, value in setup.items():
            if key == "oversample":
                if not isinstance(value, int) or value < 1:
                    raise ValueError("Oversample must be an integer >= 1")
            elif key in ["t_sim", "fps", "slowdown"]:
                if isinstance(value, int):
                    setup[key] = float(value)
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{key} must be a positive float")
                if setup[key] <= 0.:
                    raise ValueError(f"{key} must be positive")
        
        return
    
    def validate_input(self, params, required, label):
        missing_keys = [key for key in required if key not in params]
        if missing_keys:
            raise ValueError(f"Missing required keys in {label:s}: {missing_keys}")
        
        for key, value in params.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
        
        return


    def solve_ode(self, atol=1e-8, rtol=1e-8):

        start = perf_counter()
        sol = odeint(self.dynamics, self.U0, self.full_t, tfirst=True, atol=atol, rtol=rtol).T
        # sol = solve_ivp(
        #     self.dynamics, [0, self.t_sim], self.U0, 
        #     method="LSODA", t_eval=self.full_t, 
        #     atol=atol, rtol=rtol
        # ).y
        end = perf_counter()
        print(f"\tElapsed time : {end-start:.3f} seconds")
        
        self.full_series = sol
        self.series = self.full_series[:, ::self.oversample]
        
        self.full_kinematics = self.compute_kinematics()
        self.kinematics = self.full_kinematics[:, ::self.oversample]
        
        return

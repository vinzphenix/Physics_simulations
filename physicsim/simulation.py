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


    def solve_ode(self, atol=1e-8, rtol=1e-8, verbose=True):

        start = perf_counter()
        sol = odeint(self.dynamics, self.U0, self.full_t, tfirst=True, atol=atol, rtol=rtol).T
        # sol = solve_ivp(
        #     self.dynamics, [0, self.t_sim], self.U0, 
        #     method="LSODA", t_eval=self.full_t, 
        #     atol=atol, rtol=rtol
        # ).y
        end = perf_counter()
        if verbose:
            print(f"\tElapsed time : {end-start:.3f} seconds")
        
        self.full_series = sol
        self.series = self.full_series[:, ::self.oversample]
        
        self.full_kinematics = self.compute_kinematics()
        self.kinematics = self.full_kinematics[:, ::self.oversample]
        
        return

    def wrap_angles(self, *angles_idxs):
        for idx in angles_idxs:
            phi = self.full_series[idx]
            phi[:] = np.remainder(phi[:] + np.pi, 2 * np.pi) - np.pi
        return

    def get_cut_series(self, *angles_idxs):
        self.wrap_angles(*angles_idxs)
        n2, n3 = self.full_series.shape[0], self.full_kinematics.shape[0]
        all_series = np.c_[self.full_t, self.full_series.T, self.full_kinematics.T].T
        new_series = [this_series.copy() for this_series in all_series]
        # insert np.nan to break the line when the angle goes from -pi to pi
        for idx in angles_idxs:
            phi = new_series[1+idx]
            jump_idxs = np.where(np.abs(np.diff(phi)) > 3)[0] + 1
            if len(jump_idxs) == 0:
                continue
            # print(idx, jump_idxs)
            for k in range(len(new_series)):
                # print(k)
                # print(new_series[k])
                if k == idx+1:  # insert np.nan in the angle
                    new_series[k] = np.insert(new_series[k], jump_idxs, np.nan)
                else:  # insert the same value as before the jump
                    new_series[k] = np.insert(new_series[k], jump_idxs, new_series[k][jump_idxs])
                # print(new_series[k])
        return new_series[0], new_series[1:1+n2], new_series[1+n2:]
    

def countDigits(a):
    if a < 0.:
        return max(2, 2 + int(np.log10(np.abs(a))))
    elif a > 0.:
        return max(1, 1 + int(np.log10(np.abs(a))))
    else:
        return 1
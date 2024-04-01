import physicsim.pendulum_1 as pendulum_1
import numpy as np


if __name__ == "__main__":

    setup = {"t_sim": 10., "fps": 30, "slowdown": 1., "oversample": 1}
    params = {"g": 9.81, "D": 0., "m": 1., "l": 1.}
    initials = {"phi": np.radians(170.), "om": 0.}

    sim = pendulum_1.Pendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

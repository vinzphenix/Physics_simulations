import physicsim.pendulum_foucault as foucault
import numpy as np

if __name__ == "__main__":

    params = {'g': 9.81, 'L': 100, 'phi': np.radians(75), 'W': 100.0}
    initials = {'alpha': np.radians(80), 'dalpha': 0, 'beta': 0., 'dbeta': 0.}
    # initials = {'alpha': 20, 'dalpha': 0, 'beta': 90, 'dbeta': 180}

    setup = {'t_sim': 100, 'fps': 30, 'slowdown': 1.0, 'oversample': 10}

    sim = foucault.FoucaultPendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

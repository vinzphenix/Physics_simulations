import physicsim.double_disk as db_disk
import numpy as np
from utils.display import see_path
from physicsim.simulation import countDigits

def display(sim):
    
    t = sim.full_t
    th, om, x, dx = sim.full_series
    phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2 = sim.full_kinematics

    params1 = np.array([sim.R, sim.M, sim.a, sim.m])
    params2 = np.array([sim.thd, sim.om, sim.x, sim.dx])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $\vartheta$",
        r"Axe y : $\omega$",
        r"Axe c : $\dot x$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$R \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.R, width=fmt1, dcm=dcm1),
        r"$M \;\:\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.M, width=fmt1, dcm=dcm1),
        r"$a \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.a, width=fmt1, dcm=dcm1),
        r"$m \;\:\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\vartheta$ = {:>{width}.{dcm}f} $\rm ^\circ$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt2, dcm=dcm2),
        r"$x$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt2, dcm=dcm2),
        r"$\dot x$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dx, width=fmt2, dcm=dcm2),
    ]

    see_path(th, dx, dx, var_case=2, save="", displayedInfo=parameters)
    return


if __name__ == "__main__":

    params = {
        'g': 9.81, 'k': -0.00, 'k1': 0.00,
        'R': 0.50, 'M': 1.00, 'a': 0.20, 'm': 1.00, 
    }

    initials = {
        'th': np.radians(-80.00), 'om': 0 * np.sqrt(params['g'] / params['R']), 
        'x': 0.00, 'dx': 0.50,
    }

    setup = {
        't_sim': 10., 'fps': 30, 'slowdown': 1.0, 'oversample': 10,
    }

    sim = db_disk.NestedDisks(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

    # display(sim)

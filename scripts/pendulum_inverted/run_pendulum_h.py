import physicsim.pendulum_horiztontal as h_pendulum
import numpy as np
from physicsim.simulation import countDigits
from utils.display import see_path
from numpy import pi, sqrt


def load_configuration(i):
    g = 9.81
    if 1 <= i <= 3:
        l, mp, mb, w = 0.5, 0.001, 2, 2 * pi * 10
        x, dx, phi, om = 0, 0, pi / 4, 0
        F_list = [200, 599, 802.95]
        F = F_list[i-1]
    elif i == 4:
        l, mp, mb, F, w = 0.4, 1.00, 10.0, 50, sqrt(g / (0.4) ** 2)
        x, dx, phi, om = 0, 0, pi / 4, 0  # [om,cos(phi)]
    elif i == 5:
        l, mp, mb, F, w = 0.4, 2.5, 10, 50, 2.5 * sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0
    elif i == 6:
        l, mp, mb, F, w = 0.5, 0.001, 10, 1000, 10 * g / 0.4
        x, dx, phi, om = 0, 0, pi / 3, 0
    elif i == 7:
        l, mp, mb, F, w = 1.0, 1.0, 5.0, 2000, 20 * sqrt(g/1.0)
        x, dx, phi, om = 0, 0, pi / 12, 0
    elif i == 8:
        l, mp, mb, F, w = 0.3, 10.0, sqrt(g / 0.3), 80, 4 * sqrt(g / 0.3 ** 2)
        x, dx, phi, om = 0, 0, 0.39 * pi, 0
    elif i == 9:
        l, mp, mb, F, w = 0.4, 10.0, 2.5, 50, sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0
    elif i == 10:
        l, mp, mb, F, w = 0.4, 10.0, 2.5, 50, sqrt(g)
        x, dx, phi, om = 0, 0, 0., 0
    elif i == 11:
        l, mp, mb, F, w = 1.0, 1.0, 5.0, 0., 50*sqrt(g)
        x, dx, phi, om = 0, 0, pi, 3.0
    elif i == 12:
        l, mp, mb, F, w = 0.33, 10., 3.0, 80, 40
        x, dx, phi, om = 0, 0, pi/2, 0.0
    elif i == 13:
        l, mp, mb, F, w = 0.4, 10.0, 6.0, 81, 30.
        x, dx, phi, om = 0, 0, 0.6*pi, 0
    elif i == 14:
        l, mp, mb, F, w = 0.4, 2.5, 10, 50, 2.5 * sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0

    params = {'g': g, 'l': l, 'mp': mp, 'mb': mb, 'F': F, 'w': w}
    initials = {'x': x, 'dx': dx, 'phi': phi, 'om': om}
    return params, initials
    

def display(sim):
    t = sim.full_t
    x, dx, phi, om = sim.full_series
    xb, yb, xp, yp, vp = sim.full_kinematics

    params1 = np.array([sim.l, sim.mp, sim.mb])
    params2 = np.array([sim.g, sim.F, sim.w])
    params3 = np.array([sim.x, sim.dx, sim.phid, sim.om])

    dcm1, dcm2, dcm3 = 3, 3, 3
    fmt1, fmt2, fmt3 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2, 1 + 1 + dcm3
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)
    for val in params3:
        fmt3 = max(fmt3, countDigits(val) + 1 + dcm3)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l \quad$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l, width=fmt1, dcm=dcm1),
        r"$m_p$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.mp, width=fmt1, dcm=dcm1),
        r"$m_b$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.mb, width=fmt1, dcm=dcm1),
        "",
        r"$g\,$ = {:>{width}.{dcm}f} $\rm m/s^2$".format(sim.g, width=fmt2, dcm=dcm2),
        r"$F$ = {:>{width}.{dcm}f} $\rm N$".format(sim.F, width=fmt2, dcm=dcm2),
        r"$w$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.w, width=fmt2, dcm=dcm2),
        "",
        r"$x \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt3, dcm=dcm3),
        r"$dx$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dx, width=fmt3, dcm=dcm3),
        r"$\varphi \,\,\,$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phid, width=fmt3, dcm=dcm3),
        r"$\omega \,\,\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt3, dcm=dcm3)
    ]
    parameters[0] = r"Axe x : $x_p^2$"
    parameters[1] = r"Axe y : $y_p$"
    parameters[2] = r"Axe c : $v_p$"

    see_path(xp, yp, vp, colors='jet', var_case=1, name='0', save="no", displayedInfo=parameters)
        
    return


if __name__ == "__main__":

    params = {
        'g': 9.81, 'l': 0.4, 'mp': 10.0, 
        'mb': 10.0, 'F': 50.0, 'w': 2.5 * sqrt(9.81),
    }

    initials = {
        'x': 0.0, 'dx': 0.0, 'phi': 3 * pi / 4, 'om': 0.0,
    }

    setup = {
        't_sim': 150.0, 'fps': 30, 'slowdown': 3.0, 'oversample': 50,
    }

    params, initials = load_configuration(7)

    sim = h_pendulum.HorizontalPendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

    # display(sim)

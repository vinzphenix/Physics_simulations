import physicsim.cylinder_slide as fcyl
from utils.display import countDigits, see_path_1, see_path
import numpy as np

def display(sim):

    t = sim.full_t
    th, om, x, dx = sim.full_series
    xc, yc, x1, y1, x2, y2, x3, y3, vx, vy, v2 = sim.full_kinematics

    decimals = 3
    fmt = 1 + 1 + decimals
    for val in list(sim.params.values()) + list(sim.initials.values()):
        fmt = max(fmt, countDigits(val) + 1 + decimals)

    parameters = [
        r"Axe x : $x \;*\; \omega$",
        r"Axe y : $x \;*\; v$",
        r"Axe c : $\vartheta$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(sim.t_sim), "",
        r"$\alpha_{{slope}} \,\:$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.alpha, width=fmt, dcm=decimals),
        r"$r_{{wheel}} \:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.R, width=fmt, dcm=decimals),
        r"$m_{{wheel}} $ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m, width=fmt, dcm=decimals),
        r"$m_{{pdlm}} \,\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.M, width=fmt, dcm=decimals),
        r"$L_{{pdlm}} \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(sim.L, width=fmt, dcm=decimals),
        "", r"$C_{{wheel}}$ = {:>6.3f} $\rm N \, m$".format(sim.C), "",
        r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\vartheta$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.thd, width=fmt, dcm=decimals),
        r"$\omega$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt, dcm=decimals),
        r"$x$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt, dcm=decimals),
        r"$\dot x$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dx, width=fmt, dcm=decimals)
    ]
    parameters[0] = r"Axe x : $\omega$"
    parameters[1] = r"Axe y : $v$"
    parameters[2] = r"Axe c : $\vartheta$"

    s = np.s_[:245:-1]
    see_path_1(
        2, np.array([om[s], dx[s]]), th[s], color='Blues', shift=(0.15, -0.15),
        var_case=2, save="no", displayedInfo=parameters
    )
    return


def load_configuration(i):

    g, D1, D2 = 9.81, 0., 0.
    # 64 seconds
    if i == 1:
        alpha, C = np.radians(10.0), -8.12
        R, m = 0.80, 1.0
        L, M = 1.0, 5.0
        th, om, x, v = np.radians(170.), 0.00, 0.00, 0.00
    else:
        raise ValueError("Invalid configuration number")

    params = {
        'g': g, 'alpha': alpha, 
        'R': R, 'm': m, 'L': L, 'M': M, 
        'C': C, 'D1': D1, 'D2': D2
    }
    initials = {
        'th': th, 'om': om, 'x': x, 'dx': v
    }
    return params, initials


if __name__ == "__main__":

    prm = {
        'g': 9.81, 'alpha': np.radians(10.0),
        'R': 0.80, 'm': 10.0, 'L': 1.00, 'M': 1.0,
        'C': -15., 'D1': 0.0, 'D2': 0.0
    }
    torque_eq = -(prm['m'] + prm['M']) * prm['g'] * prm['R'] * np.sin(prm['alpha'])
    # print(torque_eq)
    initials = {
        'th': np.radians(179.), 'om': -0.10, 'x': 0.00, 'dx': -1.00
    }
    setup = {
        "t_sim": 10.0, "fps": 30, "slowdown": 1., "oversample": 10
    }

    prm, initials = load_configuration(1)

    sim = fcyl.Cylinder(setup, params=prm, initials=initials)
    sim.solve_ode()
    sim.animate(save="no")

    # display(sim)

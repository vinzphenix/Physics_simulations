import physicsim.pendulum_L as pendulum_L
from physicsim.simulation import countDigits
from utils.display import see_path
import numpy as np

def display(sim):
    t = sim.full_t
    th, phi1, phi2, om, om1, om2 = sim.full_series
    x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y = sim.full_kinematics

    dcm1, dcm2 = 2, 3
    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$L - \,h\, - \,m\,\:$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.L, sim.h, sim.M, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        r"$l_1 - h_1 - m_1$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.l1, sim.h1, sim.m1, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        r"$l_2 - h_2 - m_2$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.l2, sim.h2, sim.m2, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        "",
        r"$g\,$ = {:.2f} $\rm m/s^2$".format(sim.g),
        "",
        r"$\vartheta \:\;- \omega\:\:$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.th, sim.om, dcmA=dcm2, dcmB=dcm2),
        r"$\varphi_1 - \omega_1$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.phi1, sim.om1, dcmA=dcm2, dcmB=dcm2),
        r"$\varphi_2 - \omega_2$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.phi2, sim.om2, dcmA=dcm2, dcmB=dcm2)
    ]
    parameters[0] = r"Axe x : $x_p$"
    parameters[1] = r"Axe y : $y_p$"
    parameters[2] = r"Axe c : $v_p$"

    see_path(
        th, phi1, np.hypot(v2x, v1y), 
        colors='jet', save="no", displayedInfo=parameters
    )

    see_path(
        [th, phi1, phi2], [om, om1, om2], 
        [np.hypot(vx, vy), np.hypot(v1x, v1y), np.hypot(v2x, v2y)],
        colors=['viridis', 'inferno_r', 'jet'], var_case=2, save=False
    )
    return

if __name__ == "__main__":

    sqrt2 = np.sqrt(2)
    params = {
        "g": 9.81,
        "L": 0.75, "h": 0.1, "M": 10.0, "D": 0.,
        "l1": 0.75, "h1": 0.1, "m1": 1., "D1": 0.,
        "l2": 0.75, "h2": 0.1, "m2": 1., "D2": 0.
    }
    initials = {
        "th": np.radians(0), 
        "phi1": -np.radians(34.),
        "phi2": np.radians(34),
        "om": np.radians(0.),
        "om1":-np.radians(0.), 
        "om2": np.radians(8.)
    }
    setup = {"t_sim": 60., "fps": 50., "slowdown": 0.5, "oversample": 50}

    sim = pendulum_L.PendulumL(setup, params, initials)
    sim.solve_ode(atol=1.e-9, rtol=1.e-9)
    sim.animate(show_v=False, show_a=False, phaseSpace=0, save="no")

    # display(sim)

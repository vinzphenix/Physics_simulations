import physicsim.pendulum_2 as db_pendulum
from utils.display import see_path
import numpy as np
from numpy import pi, sqrt

def load_configuration(i):
    # init_data = phi1, om1, phi2, om2

    if i == 1:
        l, mu = 0.25, 0.75
        init_data = 0.000, 0.732, -0.017, -0.136

    elif i == 2:
        l, mu = 0.5, 0.1
        init_data = 0.00000, 0.12762, -2.49751, -0.23353

    elif i == 3:
        l, mu = 0.5, 0.5
        init_data = 0.000, 0.901, 0.571, 0.262

    elif i == 4:
        l, mu = 0.8, 0.1
        init_data = 0.00631, 0.03685, 0.00000, 2.85528

    elif 5 <= i <= 8:
        l, mu = 1., 0.05
        u_zero_list = [
            [2 * pi / 9, 0, 2 * pi / 9 * sqrt(2), 0],  # loop
            [0.000, 0.304, -0.209, -3.160],
            [0.000, 0.617, 1.488, -0.791],
            [0.000, 0.588, 1.450, -1.223],
        ]
        init_data = u_zero_list[i - 5]
        # [0.000, 0.642, -0.485, 0.796],  # same loop trajectory, different initial conditions
        # [-0.31876, 0.54539, 0., 1.08226],  
        # [-0.68847079, -0.00158915, -0.9887163, 0.00234737],

    elif i == 9:
        l, mu = 1., 0.2
        init_data = -0.72788, -0.47704, 0.00000, 1.30007

    elif i == 10:
        l, mu = 1., 0.5
        init_data = -0.707, 0.091, 0.000, 0.611

    elif 11 <= i <= 14:
        l, mu = 2, 0.1
        u_zero_list = [
            [0.000, 0.84037, 1.72621, 0.97821],
            [0.000, 0.93306, -1.09668, 0.82402],
            [0.000, 0.92070, -2.35338, 0.14850],
            [0.000, 0.86758, -0.00314, 1.00024],
        ]
        init_data = u_zero_list[i - 11]

    elif i == 15:
        l, mu = 2., 0.3
        init_data = 0.00000, 1.32395, -1.22116, -0.09790

    elif 16 <= i <= 18:
        l, mu = 2., 0.5
        u_zero_list = [
            [0.000, 1.312, 0.840, -0.431],
            [0.000, 1.723, 1.145, -0.588],
            [0.000, 1.865, 0.008, 0.131],
        ]
        init_data = u_zero_list[i - 16]

    elif i == 19:
        l, mu = 2., 2./3.
        init_data = -1.11776, 0.85754, 0.00000, 0.80942

    elif i == 20:
        l, mu = 2., 0.8
        init_data = -0.00811, -1.28206, 0.00000, 0.93208

    elif 21 <= i <= 24:
        l, mu = 3., 1. / 3.
        u_zero_list = [
            [0.000, 1.832, -0.622, -0.550],
            [0.000, 1.644, 0.841, -0.466],
            [0.000, 1.974, 0.832, -0.558],
            [0.00000, 0.27063, -1.89320, -0.84616],
        ]
        init_data = u_zero_list[i - 21]

    elif 25 <= i <= 29:
        l, mu = 3., 0.5
        u_zero_list = [
            [0.00000, 3.18131, -0.99358, -0.65052],
            [0.00000, 3.20265, 2.65215, 0.11353],
            [0.00000, 3.50990, -2.18652, 0.65160],
            [0.00000, 3.08546, 3.14036, 0.05781],
            [0.00000, 5.12663, 2.54620, -0.54259],
        ]
        init_data = u_zero_list[i - 25]

    elif i == 30:
        l, mu = 3., 0.9
        init_data = 0.000, 3.156, 0.004, -1.074

    elif 31 <= i <= 33:
        l, mu = 5., 0.2
        u_zero_list = [
            [0.00000, 0.87736, -1.67453, -0.05226],
            [0.00000, 0.90244, 0.43123, 0.48978],
            [0.00000, 2.67628, 0.01220, -0.30299],
        ]
        init_data = u_zero_list[i - 31]

    elif i == 34:
        l, mu = 1. / 3., 0.5
        init_data = 0.00000, 0.18331, -3.01259, -4.32727
    
    elif i == 35:
        l, mu = 1. / 3., 0.5
        init_data = 0.00000, 0.24415, -2.77591, 0.05822

    elif i == 36:
        l, mu = 1., 0.1
        init_data = -0.70048, 0.92888, 0.00000, 1.96695

    elif i == 37:
        l, mu = 1., 0.9
        init_data = 0.00000, 1.24989, 1.23706, -1.05777
        # init_data = 0.00000, 1.24501, -1.25257, -1.01287  # bit more fuzzy

    elif i == 38:
        l, mu = 1., 1. / (2 + 1.)
        init_data = [0.000, np.sqrt(1/9.81), 0.000, 8.5*np.sqrt(1/9.81)]

    elif i == 39:
        l, mu = 1. / 2., 1. / (1 + 1.75)
        init_data = [0.000, -1.749 * np.sqrt(2/9.81), 0.000, 1.749 * np.sqrt(2/9.81)]

    elif i == 40:
        l, mu = 1., 1. / (1 + 10)
        init_data = [0.000, 3.5 * np.sqrt(1/9.81), 0.000, -4 * np.sqrt(2/9.81)]

    elif i == 41:
        l, mu = 1., 2. / (2. + 1.)  # fuzzy
        # l, mu = 1., 2.057 / (2.057 + 1.)  # sharp
        init_data = [0.000, 3.5 * np.sqrt(1/9.81), 0.000, -4 * np.sqrt(2/9.81)]

    elif i == 42:
        l, mu = 1.810, 1. / (12. + 1.)
        init_data = [0.698, 0.000, 0.987, 0.000]

    elif i == 43:
        l, mu = 1. / 1.1, 1. / (12. + 1.)
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]
    
    elif i == 44:
        l, mu = 1., 1. / (1. + 4.)  # fuzzy
        # l, mu = 1., 1. / (1. + 3.984209)  # sharp
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]

    elif i == 45:
        l, mu = np.sqrt(2), 0.5  # fuzzy
        # l, mu = 1., 0.5  # sharp
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]
    
    elif i == 46:
        l, mu = 1., 0.260
        init_data = [0.00000, 0.76537, 0.00000, -0.76537]

    elif i == 47:
        l, mu = 0.5, 1. / (1. + 1.75)
        init_data = [0.00000, -0.78971, 0.00000, 0.78971]

    elif 48 <= i <= 53:
        l, mu = 1., 0.5
        u_zero_list = [
            [0.58673, 0.00000, 2.85847, 0.00000],
            [0.87511, 0.00000, -2.05532, 0.00000],
            [1.44555, 0.00000, 2.62321, 0.00000],
            [-7./36. * np.pi, 0., 4./75. * np.pi, 0.],
            [-np.pi/6., 0., np.pi/12., 0.],
            [-7./144.*np.pi, 0., 5./18.*np.pi, 0.]
        ]
        init_data = u_zero_list[i - 48]

    elif i == 54:
        l, mu = 1., 1. / (2. + 1.)
        init_data = [0., 3.5/np.sqrt(9.81), 0.00000, -5.657/np.sqrt(9.81)]

    elif i == 55:
        l, mu = 2., 0.5
        init_data = [0.00000, 1.31200, 0.84000, -0.43100]

    elif i == 56:
        l, mu = 2., 2. / (2. + 1.)
        init_data = [2*np.pi/3, -0.54, 0.00000, 0.54]

    else:
        raise ValueError("Invalid configuration number")

    params = {'g': 9.81, 'l1': 1., 'l2': l, 'm1': 1., 'm2': 1. * mu / (1 - mu), 'adim': True}
    initials = dict(zip(['phi1', 'om1', 'phi2', 'om2'], init_data))
    return params, initials


def display(sim, wrap=False):
    if wrap:
        t, series, kinematics = sim.get_cut_series(0, 2)  # phi1, phi2
    else:
        t, series, kinematics = sim.full_t, sim.full_series, sim.full_kinematics

    phi1, om1, phi2, om2 = series
    x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2, T1, T2 = kinematics

    parameters = sim.get_parameters()

    parameters[0] = r"Axe x : $x_2$"
    parameters[1] = r"Axe y : $y_2$"
    parameters[2] = r"Axe c : $v_2$"
    # see_path(x2, y2, v2, colors='inferno', var_case=1, save="no", displayedInfo=parameters, name='1')
    # see_path(phi2, om2, v1, colors='inferno', var_case=2, save="no", displayedInfo=parameters, name='2')
    
    parameters[0] = r"Axe x : $\varphi_1$ - $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$ - $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$ - $\varphi_1$"
    see_path([phi1, phi2], [om1, om2], [v1*v2, v1*v2],
            colors=["YlOrRd", "YlGnBu"],
            # colors=["inferno", "inferno"],
            var_case=2, save="no", displayedInfo=parameters
    ) 
    return

if __name__ == "__main__":

    params = {
        'g': 9.81, 'l1': 0.50, 'l2': 0.50, 'm1': 1.75, 'm2': 1
    }

    initials = {
        'phi1': np.radians(75), 'phi2': -np.radians(50.), 'om1': 0., 'om2': 0., 
    }

    setup = {
        't_sim':50.0, 'fps': 30, 'slowdown': 0.3, 'oversample': 10
    }

    params, initials = load_configuration(26)

    sim = db_pendulum.DoublePendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(figsize=(13., 6.), show_v=False, show_a=False, wrap=True, save="no")

    # display(sim, wrap=True)

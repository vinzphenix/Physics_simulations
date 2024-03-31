import physicsim.double_pendulum as db_pendulum
from utils.display import countDigits, see_path_1, see_path
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
        time_series = sim.get_cut_series()
    else:
        time_series = np.r_[sim.full_t.reshape(1, -1), sim.full_series]

    t, phi1, om1, phi2, om2 = time_series
    x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2 = sim.full_kinematics

    params1 = np.array([sim.l1, sim.l2, sim.m1, sim.m2])
    params2 = np.array([sim.phi1d, sim.phi2d, sim.om1, sim.om2])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l1, width=fmt1, dcm=dcm1),
        r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l2, width=fmt1, dcm=dcm1),
        r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m1, width=fmt1, dcm=dcm1),
        r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m2, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi1d, width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi2d, width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2),
    ]

    parameters[0] = r"Axe x : $x_2$"
    parameters[1] = r"Axe y : $y_2$"
    parameters[2] = r"Axe c : $v_2$"
    # see_path_1(1, np.array([x2, y2]), ac2, color='viridis', var_case=1,  shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1., np.array([phi1, phi2]), v1, color='inferno', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='1')
    
    parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
    # see_path(1, [np.array([phi1, om1]), np.array([phi2, om2])],
    #         [v1*v2, v1*v2],
    #         #["YlOrRd", "YlGnBu"],
    #         ["inferno", "inferno"],
    #         [(-0., 0.), (-0., 0.)],
    #         var_case=2, save="no", displayedInfo=parameters
    # ) 
    return

if __name__ == "__main__":

    params = {
        'g': 9.81, 'l1': 0.50, 'l2': 0.50, 'm1': 1.75, 'm2': 1
    }

    initials = {
        'phi1': np.radians(75), 'phi2': -np.radians(50.), 'om1': 0., 'om2': 0., 
    }

    setup = {
        't_sim':250.0, 'fps': 30, 'slowdown': .15, 'oversample': 10
    }

    params, initials = load_configuration(55)

    sim = db_pendulum.DoublePendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(figsize=(13., 6.), show_v=False, show_a=False, wrap=False, save="no")

    # display(sim, wrap=False)

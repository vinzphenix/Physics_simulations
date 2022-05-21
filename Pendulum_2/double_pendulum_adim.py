import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, sqrt, pi, hypot
from scipy.integrate import odeint
from Utils.Fixed_Path import countDigits, see_path_1, see_path
from Utils.dialog_box import ask


def dynamics(u, _, l, mu):
    th1, w1, th2, w2 = u
    Cos, Sin = cos(th1 - th2), sin(th1 - th2)

    f1 = (mu * Cos * (sin(th2) - w1 * w1 * Sin) - l * mu * w2 * w2 * Sin - sin(th1)) / (1. - mu * Cos * Cos)
    f2 = Sin * (cos(th1) + w1 * w1 + l * mu * w2 * w2 * Cos) / (l - l * mu * Cos * Cos)

    return np.array([w1, f1, w2, f2])


def db_pendulum_solver(u_zero, l=1., mu=0.5, mode='auto', Tend=20, sps=720, l1=1., m1=1.):
    lw, cmap = 2, 'inferno'
    if mode == 'animation_ask':
        Tend, l1, m1 = ask('animation', [])
    elif mode == 'draw_ask':
        Tend, l1, m1, lw, cmap = ask('draw', [])
    Tend, l1, m1, lw = float(Tend), float(l1), float(m1), float(lw)

    n, fps = int(sps * Tend), 20
    ratio = n // (int(Tend * fps))

    g = 9.81
    w = sqrt(g / l1)
    t = np.linspace(0, Tend * w, n)
    sol = odeint(dynamics, u_zero, t, args=(l, mu))
    phi1, om1, phi2, om2 = sol.T

    t /= w
    om1, om2 = om1 * w, om2 * w
    u_zero[[1, 3]] *= w

    params = l1, l1 * l, m1, m1 * mu / (1 - mu), *u_zero
    v_params = n, fps, ratio, lw, cmap

    return t, phi1, phi2, om1, om2, params, v_params


def draw_image(_, phi1, phi2, om1, om2, params, v_params):
    l1, l2, m1, m2, phi1_0, om1_0, phi2_0, om2_0 = params
    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

    vx2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
    vy2 = l1 * om1 * sin(phi1) + l2 * om2 * sin(phi2)
    v2 = hypot(vx2, vy2)
    # sqrt(((l1 * om1) ** 2 + (l2 * om2) ** 2 + 2 * l1 * l2 * om1 * om2 * cos(phi1 - phi2)))

    n, fps, ratio, lw, cmap = v_params
    see_path_1(lw, np.array([x2, y2]), v2, cmap)


def see_animation(t, phi1, phi2, om1, om2, params, v_params, size=(8.4, 4.8), save="", inside=False):
    l1, l2, m1, m2, phi1_0, om1_0, phi2_0, om2_0 = params
    n, fps, ratio, lw, cmap = v_params
    L, Tend = l1 + l2, t[-1]
    scale = size[0] / 14

    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

    #####     ================      Création de la figure      ================      #####
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = False
    ftSz1, ftSz2, ftSz3 = 20, 17, 12

    ratio = 1 if save == "snapshot" else ratio
    # plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(121, autoscale_on=False, xlim=(-L * 1.1, L * 1.1), ylim=(-1.1 * L, 1.1 * L), aspect='equal')
    ax.grid(ls=':')
    ax2 = fig.add_subplot(122)
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi \; \rm [\:rad\:]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \;\rm [\:rad/s\:]$', fontsize=ftSz2)

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], '-', lw=1, color='grey')

    phase1, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')

    time_template = r'$t = %.1fs$'
    time_text = ax.text(0.42, 0.94, '', fontsize=ftSz2 * scale, transform=ax.transAxes)
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(l1), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(l2), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(m1), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(m2), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)

    ax.text(0.61, 0.96, r'$\varphi_1  = {:.2f} $'.format(np.degrees(phi1_0)), fontsize=ftSz3 * scale, wrap=True,
            transform=ax.transAxes)
    ax.text(0.61, 0.92, r'$\varphi_2  = {:.2f} $'.format(np.degrees(phi2_0)), fontsize=ftSz3 * scale, wrap=True,
            transform=ax.transAxes)
    ax.text(0.81, 0.96, r'$\omega_1  = {:.2f} $'.format(np.degrees(om1_0)), fontsize=ftSz3 * scale, wrap=True,
            transform=ax.transAxes)
    ax.text(0.81, 0.92, r'$\omega_2  = {:.2f} $'.format(np.degrees(om2_0)), fontsize=ftSz3 * scale, wrap=True,
            transform=ax.transAxes)

    # phi1 = np.fmod(np.fmod(phi1, 2 * pi) + 2 * pi + pi, 2 * pi) - pi
    # phi2 = np.fmod(np.fmod(phi2, 2 * pi) + 2 * pi + pi, 2 * pi) - pi
    ax2.plot(phi1, om1, color='C1', ls='-', marker='', markersize=0.1)
    ax2.plot(phi2, om2, color='C2', ls='-', marker='', markersize=0.1)

    #####     ================      Animation      ================      #####

    def init():
        for line in [line1, line2, line3]:
            line.set_data([], [])
        for phase in [phase1, phase2]:
            phase.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)

        liste = [line1, line2, line3, phase1, time_text, phase2, sector]

        return tuple(liste)

    def update(i):
        i *= ratio
        start = max(0, i - 250000)
        thisx, thisy = [0, x1[i]], [0, y1[i]]
        thisx2, thisy2 = [x1[i], x2[i]], [y1[i], y2[i]]

        line1.set_data(thisx, thisy)
        line2.set_data(thisx2, thisy2)
        line3.set_data([x2[start:i + 1]], [y2[start:i + 1]])
        phase1.set_data(phi1[i], om1[i])
        phase2.set_data(phi2[i], om2[i])

        time_text.set_text(time_template % (t[i]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        liste = [line1, line2, line3, phase1, time_text, phase2, sector]

        return tuple(liste)

    def onThisClick(event):
        if event.button == 3:
            anim.event_source.stop()
            plt.close(fig)
        return

    fig.canvas.mpl_connect('button_press_event', onThisClick)
    anim = FuncAnimation(fig, update, n // ratio,
                         interval=5 * (14 / size[0]) ** 1.75, blit=True, init_func=init, repeat_delay=3000)

    # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
    plt.tight_layout()

    if save == "save":
        anim.save('double_pendulum_1.html', fps=30)
    elif save == "gif":
        anim.save('./animations/trajectory.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        update(int(5. * n / Tend))
        fig.savefig("./figures/trajectory.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def __main__(what_to_do=0):
    ########################################################################################################
    #####     ================      Paramètres  de la simulation      ================      ####

    g, l1, m1, Tend = 9.81, 1.00, 1.00, 10.41

    # l, mu = 0.25, 0.75
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.732, -0.017, -0.136

    # l, mu = 1. / 3., 0.5
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.18331, -3.01259, -4.32727
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.24415, -2.77591, 0.05822

    # l, mu = 0.5, 0.1
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.12762, -2.49751, -0.23353

    # l, mu = 0.5, 0.5
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.901, 0.571, 0.262

    # l, mu = 0.8, 0.1
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00631, 0.03685, 0.00000, 2.85528

    l, mu = 1, 0.05
    # phi1_0, om1_0, phi2_0, om2_0 = 2 * pi / 9, 0, 2 * pi / 9 * sqrt(2), 0  # loop
    phi1_0, om1_0, phi2_0, om2_0 = -0.31876, 0.54539, 0., 1.08226  # loop
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.642, -0.485, 0.796  # loop
    # phi1_0, om1_0, phi2_0, om2_0 = -0.68847079, -0.00158915, -0.9887163, 0.00234737  # loop
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.617, 1.488, -0.791  # 1 ; 0.05 ; good
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.304, -0.209, -3.160  # 1 ; 0.05 ;
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 0.588, 1.450, -1.223  # 1 ; 0.05 ;

    # l, mu = 1., 0.1
    # phi1_0, om1_0, phi2_0, om2_0 = -0.70048, 0.92888, 0.00000, 1.96695

    # l, mu = 1., 0.2
    # phi1_0, om1_0, phi2_0, om2_0 = -0.72788, -0.47704, 0.00000, 1.30007

    # l, mu = 1., 0.5
    # phi1_0, om1_0, phi2_0, om2_0 = -0.707, 0.091, 0.000, 0.611

    # l, mu = 1., 0.9
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 1.24501, -1.25257, -1.01287
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 1.24989, 1.23706, -1.05777

    # l, mu = 2, 0.1
    # phi1_0, om1_  0, phi2_0, om2_0 = 0.00000, 0.93306, -1.09668, 0.82402
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.84037, 1.72621, 0.97821
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.92070, -2.35338, 0.14850
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.86758, -0.00314, 1.00024

    # l, mu = 2, 0.3
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 1.32395, -1.22116, -0.09790

    # l, mu = 2, 0.5
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.312, 0.840, -0.431
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.865, 0.008, 0.131
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.723, 1.145, -0.588

    # l, mu = 2, 2./3.
    # phi1_0, om1_0, phi2_0, om2_0 = -1.11776, 0.85754, 0.00000, 0.80942

    # l, mu = 2, 0.8
    # phi1_0, om1_0, phi2_0, om2_0 = -0.00811, -1.28206, 0.00000, 0.93208

    # l, mu = 3, 1. / 3.
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.832, -0.622, -0.550
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.644, 0.841, -0.466
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 1.974, 0.832, -0.558
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.27063, -1.89320, -0.84616

    # l, mu = 3, 0.5
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 3.18131, -0.99358, -0.65052
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 3.20265, 2.65215, 0.11353
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 3.50990, -2.18652, 0.65160
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 3.08546, 3.14036, 0.05781
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 5.12663, 2.54620, -0.54259

    # l, mu = 3., 0.9
    # phi1_0, om1_0, phi2_0, om2_0 = 0.000, 3.156, 0.004, -1.074

    # l, mu = 5., 0.2
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.87736, -1.67453, -0.05226
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 0.90244, 0.43123, 0.48978
    # phi1_0, om1_0, phi2_0, om2_0 = 0.00000, 2.67628, 0.01220, -0.30299

    U0 = np.array([phi1_0, om1_0, phi2_0, om2_0])
    t, phi1, phi2, om1, om2, params, v_params = db_pendulum_solver(U0, l, mu, Tend=Tend, sps=720, l1=l1, m1=m1)

    if what_to_do == 0:
        see_animation(t, phi1, phi2, om1, om2, params, v_params, (12, 6), save="gif", inside=True)
    else:
        m2, l2 = m1 * mu / (1 - mu), l1 * l
        x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
        x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

        vx1, vy1 = l1 * om1 * cos(phi1), l1 * om1 * sin(phi1)
        vx2, vy2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2), l1 * om1 * sin(phi1) + l2 * om2 * sin(phi2)
        v1, v2 = hypot(vx1, vy1), hypot(vx2, vy2)

        COS, SIN = cos(phi1 - phi2), sin(phi1 - phi2)
        dom1 = (mu * COS * (sin(phi2) - om1 * om1 * SIN) - l * mu * v2 * v2 * SIN - sin(phi1)) / (1. - mu * COS * COS)
        dom2 = SIN * (cos(phi1) + om1 * om1 + l * mu * v2 * v2 * COS) / (l - l * mu * COS * COS)
        acx2 = l1 * dom1 * cos(phi1) - l1 * om1 * om1 * sin(phi1) + l2 * dom2 * cos(phi2) - l2 * om2 * om2 * sin(phi2)
        acy2 = l1 * dom1 * sin(phi1) + l1 * om1 * om1 * cos(phi1) + l2 * dom2 * sin(phi2) + l2 * om2 * om2 * cos(phi2)
        ac2 = hypot(acx2, acy2)

        params1 = np.array([l1, l2, m1, m2])
        params2 = np.array([np.degrees(phi1_0), np.degrees(phi2_0), om1_0, om2_0])
        dcm1, dcm2 = 3, 4
        fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
        for val in params2:
            fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

        parameters = [
            r"Axe x : $x_2$",
            r"Axe y : $y_2$",
            r"Axe c : $v_2$",
            "", r"$\Delta t$ = {:.2f} $\rm s$".format(Tend), "",
            r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l1, width=fmt1, dcm=dcm1),
            r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l2, width=fmt1, dcm=dcm1),
            r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(m1, width=fmt1, dcm=dcm1),
            r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(m2, width=fmt1, dcm=dcm1),
            "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(g), "",
            r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(np.degrees(phi1_0), width=fmt2, dcm=dcm2),
            r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(np.degrees(phi2_0), width=fmt2, dcm=dcm2),
            r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om1_0, width=fmt2, dcm=dcm2),
            r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om2_0, width=fmt2, dcm=dcm2)
        ]

        parameters[0] = r"Axe x : $x_2$"
        parameters[1] = r"Axe y : $y_2$"
        parameters[2] = r"Axe c : $v_2$"
        # see_path_1(1, np.array([x2, y2]), ac2, color='jet', var_case=1,  shift=(0., 0.), save="no", displayedInfo=parameters)

        # see_path_1(1., np.array([phi1, om1]), v2, color='viridis', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='1')
        # see_path_1(1., np.array([phi1, om2]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='2')
        # see_path_1(1., np.array([phi1, phi2]), v2, color='viridis', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='3')
        # see_path_1(1., np.array([phi2, om1]), v2, color='viridis', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='4')
        # see_path_1(1., np.array([phi2, om2]), v1, color='viridis', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='5')
        # see_path_1(1., np.array([om1, om2]), v1*v2, color='viridis', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='6')
        #
        # see_path_1(1., np.array([phi1**2, om1]), v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='7')
        # see_path_1(1., np.array([phi1**2, om2]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='8')
        # see_path_1(1., np.array([om1, -phi2**2]), v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='9')
        # see_path_1(1., np.array([om2, phi2**2]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='8B')
        # see_path_1(1., np.array([phi1*phi2, om1]), v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='10')
        # see_path_1(1., np.array([phi1*phi2, om2]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='11')
        #
        # see_path_1(1., np.array([phi1*om1, phi2]), v1*v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='12')
        # see_path_1(1., np.array([phi1*om1, om2]), v1*v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='12B')
        # see_path_1(1., np.array([phi1*om1, om2*om1]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='13')
        #
        # see_path_1(1., np.array([phi1, phi2*om1]), phi2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='14')
        # see_path_1(1., np.array([phi1*om2, phi2*om1]), v1*v2, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='15')
        # see_path_1(1., np.array([om2, phi2*om1]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='16')
        # see_path_1(1., np.array([phi2*om1, om2**2]), v1*v2, color='inferno', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='17')
        #
        # see_path_1(1., np.array([phi1, om2**2]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='18')
        #
        # see_path_1(1., np.array([phi1**2*om2, phi2**2*om1]), phi1*phi2*v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='19')
        # see_path_1(1., np.array([om2, cos(phi1)]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='20')
        # see_path_1(1., np.array([sin(om1), cos(phi2)]), v1, color='viridis', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='21')
        # see_path_1(1., np.array([-(phi1*phi2+1)**2, abs(om1)]), (phi1**2)*v2, color='Blues', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='22')
        # see_path_1(1., np.array([phi1*om1, -(om2)*(om1+1)]), (om1-3)*om2, color='binary', shift=(-0., -0.), var_case=2, save="no", displayedInfo=parameters, name='23')
        #
        #
        # parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
        # parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
        # parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
        # see_path(1, [np.array([phi1, om1]), np.array([phi2, om2])],
        #          [v1*v2, v1*v2],
        #          #["YlOrRd", "YlGnBu"],
        #          ["inferno", "inferno"],
        #          [(-0., 0.), (-0., 0.)],
        #           var_case=2, save="no", displayedInfo=parameters)


if __name__ == "__main__":
    __main__()

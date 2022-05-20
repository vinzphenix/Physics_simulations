import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import Utils.Fixed_Path as FxPath

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow
from numpy import sin, cos, pi, sqrt, hypot, degrees
from scipy.integrate import odeint
from timeit import default_timer as timer

# plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family'] = 'serif'
# plasma, inferno, viridis, cividis, Blues,  jet, YlOrRd, YlGnBu

initial_conditions = [
    [0, 0, 1, 8.5, 1.00, 1.00, 2.00, 1],
    [0, 0, 3.5, -4 * sqrt(2), 1.00, 1.00, 2.00, 1],
    [0, 0, 3.5, -4 * sqrt(2), 1.00, 1.00, 10.0, 1],
    [0, 0, 3.5, -4 * sqrt(2), 1.00, 1.00, 1.00, 2.],
    [0, 0, 3.5, -4 * sqrt(2), 1.00, 1.00, 1.00, 2.057],
    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.1, 1.00, 12.0, 1],
    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.00, 1.81, 12.0, 1],

    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.00, 1.00, 4.0, 1],
    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.00, 1.00, 3.984209, 1],
    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.00, 1.00, 1.00, 1],
    [2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1 / sqrt(2), 1.00, 1.00, 1],
    [0, 0, 2.3972, -2.3972, 1.00, 1.00, 2.84, 1],

    [0, 0, -1.749, 1.749, 2.00, 1.00, 1.75, 1],
    [0.5867344859865208, 2.858474925650023, 0, 0, 1.00, 1.00, 1.00, 1],
    [0.8751064890716895, -2.055322601709835, 0, 0, 1.00, 1, 1.00, 1],
    [1.4455480358016313, 2.623212807699145, 0, 0, 1.00, 1, 1.00, 1],
    [0.0, 0.84, 4.109304641907193, -1.3499316316021341, 1.0, 2.0, 1.0, 1.0],

    [np.radians(118.), 0., -1.73, 1.71, 1, 2, 1, 2]
]


def dynamics(u, _):
    th1, w1, th2, w2 = u
    Cos, Sin = cos(th1 - th2), sin(th1 - th2)
    # f1 = (m2 * Cos * (g * sin(th2) - l1 * w1 * w1 * Sin ) - m2 * l2 * w2 * w2 * Sin - M * g * sin(th1) - D*l1*w1) / (
    #        M * l1 - l1 * m2 * Cos ** 2)
    # f2 = 1 / (l2 * m1 * m2 * (-M + m2 * Cos ** 2)) * \
    #     (D * (l1 * m2 * w1 * Cos - l2 * m1 * w2) * (-M + m2 * Cos ** 2) - m1 * m2 * M * (g * cos(th1) + l1 * w1 * w1) * Sin
    #      + m2 * m2 * (D * l1 * Sin * w1 - l2 * m1 * w2 * w2) * Sin * Cos)

    # f1 = (m2 * Cos * (g * sin(th2) - l1 * w1 * w1 * Sin ) - m2 * l2 * w2 * w2 * Sin - M * g * sin(th1)) / (
    #        M * l1 - m2 * l1 * Cos ** 2)
    # f2 = Sin * ( g * M * cos(th1) + M * l1 * w1*w1 + m2 * l2 * w2*w2 * Cos) / (M * l2 - m2 * l2 * Cos ** 2)

    f1 = (-c3 * c3 * w1 * w1 * Cos * Sin + c3 * c5 * Cos * sin(
        th2) - 2 * c2 * c3 * w2 * w2 * Sin - 2 * c2 * c4 * sin(th1)) / (4 * c1 * c2 - c3 * c3 * Cos * Cos)

    f2 = (c3 * c3 * w2 * w2 * Cos * Sin + c3 * c4 * Cos * sin(
        th1) + 2 * c1 * c3 * w1 * w1 * Sin - 2 * c1 * c5 * sin(th2)) / (4 * c1 * c2 - c3 * c3 * Cos * Cos)

    return np.array([w1, f1, w2, f2])


if __name__ == "__main__":
    ########################################################################################################
    ###########     ================      Paramètres de la simulation      ================      ###########

    g = 9.81  # [m/sf]  -  accélération de pesanteur
    l1 = 1.00  # [m]     -  longueur 1er pendule
    l2 = 1.00  # [m]     -  longueur 2eme pendule
    m1 = 1.00  # [kg]    -  masse 1er pendule
    m2 = 1.00  # [kg]    -  masse 2em pendule
    D = 0.

    distributed = False
    a1, a2 = l1 / 10, l2 / 10
    I1 = m1 * (a1 * a1 + l1 * l1) / 12.
    I2 = m2 * (a2 * a2 + l2 * l2) / 12.

    phi1_0 = 80  # -40.5081  # -8.75 # -37 # -33 # -35              # [°]     -  angle 1er pendule
    om1_0 = 0.  # 0.0910                 # [rad/s] -  v angulaire 1er pendule
    phi2_0 = 69.  # 0.  # 50 #9.5# 14 # 12      # [°]     -  angle 2e pendule
    om2_0 = 0.  # 0.6110                 # [rad/s] -  v angulaire 2em pendule

    phi1_0, phi2_0 = np.radians([phi1_0, phi2_0])
    # phi1_0, om1_0, phi2_0, om2_0 = radians([phi1_0, om1_0, phi2_0, om2_0])
    # phi1_0, om1_0, phi2_0, om2_0 = np.radians(phi1_0), om1_0*sqrt(g/l1), np.radians(phi2_0), om2_0*sqrt(g/l1)

    # phi1_0, phi2_0, om1_0, om2_0, l1, l2, m1, m2 = *initial_conditions[4]

    Tend = 30.
    n, fps = int(1000 * Tend), 30
    ratio = n // (int(Tend * fps))
    lw, savefig = 1., "no"

    ########################################################################################################
    ###############     ================      Informations Figure      ================      ###############

    params1 = np.array([l1, l2, m1, m2])
    params2 = np.array([np.degrees(phi1_0), np.degrees(phi2_0), om1_0, om2_0])
    dcm1, dcm2 = 3, 3
    fmt1, fmt2 = FxPath.countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, FxPath.countDigits(val) + 1 + dcm2)

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
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi1_0), width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi2_0), width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om1_0, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om2_0, width=fmt2, dcm=dcm2)
    ]

    ########################################################################################################
    #####     ================      Résolution de l'équation différentielle      ================      #####

    L, M = l1 + l2, m1 + m2
    if distributed:
        c1 = m1 * l1 * l1 / 8. + I1 / 2. + m2 * l1 * l1 / 2.
        c2 = m2 * l2 * l2 / 8. + I2 / 2.
        c3 = m2 * l1 * l2 / 2.
        c4 = g * l1 * (m1 / 2. + m2)
        c5 = g * l2 * m2 / 2.
    else:
        c1 = (m1 + m2) * l1 * l1 / 2.
        c2 = m2 * l2 * l2 / 2.
        c3 = m2 * l1 * l2
        c4 = g * l1 * (m1 + m2)
        c5 = g * l2 * m2

    tic = timer()
    U0 = np.array([phi1_0, om1_0, phi2_0, om2_0])
    t = np.linspace(0, Tend, n)
    sol = odeint(dynamics, U0, t)
    print("      Elapsed time : %f seconds" % (timer() - tic))

    phi1, om1, phi2, om2 = sol.T

    COS, SIN = cos(phi1 - phi2), sin(phi1 - phi2)

    dom1 = (m2 * COS * (g * sin(phi2) - l1 * om1 * om1 * SIN) - m2 * l2 * om2 * om2 * SIN - M * g * sin(phi1)) / (
        M * l1 - m2 * l1 * COS ** 2)
    dom2 = SIN * (g * M * cos(phi1) + M * l1 * om1 * om1 + m2 * l2 * om2 * om2 * COS) / (M * l2 - m2 * l2 * COS ** 2)

    ########################################################################################################
    ########     ================      Écriture des Positions / Vitesses      ================      ########

    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

    vx1, vy1 = l1 * om1 * cos(phi1), l1 * om1 * sin(phi1)
    vx2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
    vy2 = l1 * om1 * sin(phi1) + l2 * om2 * sin(phi2)
    v1, v2 = hypot(vx1, vx2), hypot(vx2, vy2)

    acx2 = l1 * dom1 * cos(phi1) - l1 * om1 * om1 * sin(phi1) + l2 * dom2 * cos(phi2) - l2 * om2 * om2 * sin(phi2)
    acy2 = l1 * dom1 * sin(phi1) + l1 * om1 * om1 * cos(phi1) + l2 * dom2 * sin(phi2) + l2 * om2 * om2 * cos(phi2)
    ac2 = hypot(acx2, acy2)

    vx_max, vy_max = np.amax(vx2), np.amax(vy2)
    ax_max, ay_max = np.amax(acx2), np.amax(acy2)

    ########################################################################################################
    ###############     ================      Animation du système      ================      ##############

    def see_animation(arrow_velocity=False, arrow_acceleration=False, save=False):
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            phase1.set_data([], [])
            phase2.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            liste = [line1, line2, line3, phase1, time_text, phase2, sector]
            if arrow_velocity:
                nonlocal a_s
                a_s = axs[0].add_patch(arrow_s)
                liste.append(arrow_s)
            if arrow_acceleration:
                nonlocal a_a
                a_a = axs[0].add_patch(arrow_a)
                liste.append(arrow_a)
            return tuple(liste)

        def animate(i):
            i *= ratio
            start = max(0, i - 250000)
            thisx, thisx2 = [0, x1[i]], [x1[i], x2[i]]
            thisy, thisy2 = [0, y1[i]], [y1[i], y2[i]]

            line1.set_data(thisx, thisy)
            line2.set_data(thisx2, thisy2)
            line3.set_data([x2[start:i + 1]], [y2[start:i + 1]])
            phase1.set_data(phi1[i], om1[i])
            phase2.set_data(phi2[i], om2[i])

            time_text.set_text(time_template % (t[i]))
            sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
            axs[0].add_patch(sector)
            liste = [line1, line2, line3, phase1, time_text, phase2, sector]

            if arrow_velocity:
                nonlocal a_s, arrow_s
                axs[0].patches.remove(a_s)
                arrow_s = Arrow(x2[i], y2[i], vx2[i] * 2 * L / (3 * vx_max), vy2[i] * 2 * L / (3 * vy_max), color='C3',
                                edgecolor=None, width=L / 10)
                a_s = axs[0].add_patch(arrow_s)
                liste.append(arrow_s)
            if arrow_acceleration:
                nonlocal a_a, arrow_a
                axs[0].patches.remove(a_a)
                arrow_a = Arrow(x2[i], y2[i], acx2[i] * 2 * L / (3 * ax_max), acy2[i] * 2 * L / (3 * ay_max),
                                color='C4',
                                edgecolor=None, width=L / 10)
                a_a = axs[0].add_patch(arrow_a)
                liste.append(arrow_a)
            return tuple(liste)

        a_s, a_a = None, None

        #####     ================      Création de la figure      ================      #####
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        axs[0].axis([-L * 1.1, L * 1.1, -1.1 * L, 1.1 * L])
        axs[0].set_aspect("equal")
        axs[0].grid(ls=':')
        axs[1].grid(ls=':')
        axs[1].set_xlabel(r'$\varphi \; \rm [\:rad\:]$')
        axs[1].set_ylabel(r'$\omega \;\rm [\:rad/s\:]$')

        line1, = axs[0].plot([], [], 'o-', lw=2, color='C1')
        line2, = axs[0].plot([], [], 'o-', lw=2, color='C2')
        line3, = axs[0].plot([], [], '-', lw=1, color='grey')
        phase1, = axs[1].plot([], [], marker='o', ms=8, color='C0')
        phase2, = axs[1].plot([], [], marker='o', ms=8, color='C0')

        if arrow_velocity:
            dx, dy = vx2[0] * 2 * L / (3 * vx_max), vy2[0] * 2 * L / (3 * vy_max)
            arrow_s = Arrow(x2[0], y2[0], dx, dy, color='C3', edgecolor=None, width=L / 10)
            # a_s = axs[0].add_patch(arrow_s)
        if arrow_acceleration:
            dx, dy = acx2[0] * 2 * L / (3 * ax_max), acy2[0] * 2 * L / (3 * ay_max)
            arrow_a = Arrow(x2[0], y2[0], dx, dy, color='C4', edgecolor=None, width=L / 10)
            # a_a = axs[0].add_patch(arrow_a)

        time_template = r'$t = %.1fs$'
        time_text = axs[0].text(0.42, 0.94, '', fontsize=15, transform=axs[0].transAxes)
        sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')
        names = [r'$l_1  = {:.2f} \: \rm m$'.format(l1), r'$l_2  = {:.2f} \: \rm m$'.format(l2),
                 r'$m_1  = {:.2f} \: \rm kg$'.format(m1), r'$m_2  = {:.2f} \: \rm kg$'.format(m2)]
        axs[0].text(0.02, 0.96, names[0], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.02, 0.92, names[1], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.18, 0.96, names[2], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.18, 0.92, names[3], fontsize=12, wrap=True, transform=axs[0].transAxes)

        names = [r'$\varphi_1  = {:.2f} $'.format(degrees(phi1_0)), r'$\varphi_2  = {:.2f} $'.format(degrees(phi2_0)),
                 r'$\omega_1  = {:.2f} $'.format(degrees(om1_0)), r'$\omega_2  = {:.2f} $'.format(degrees(om2_0))]
        axs[0].text(0.61, 0.96, names[0], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.61, 0.92, names[1], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.81, 0.96, names[2], fontsize=12, wrap=True, transform=axs[0].transAxes)
        axs[0].text(0.81, 0.92, names[3], fontsize=12, wrap=True, transform=axs[0].transAxes)

        axs[1].plot(phi1, om1, color='C1')
        axs[1].plot(phi2, om2, color='C2')

        #####     ================      Animation      ================      #####
        anim = FuncAnimation(fig, animate, n // ratio, interval=5, blit=True, init_func=init, repeat_delay=3000)
        # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
        if save:
            anim.save('double_pendulum_1.html', fps=30)
        else:
            plt.show()


    # variables, color_var = [np.array([np.log(phi1 ** 2 + 1), phi2])], v2 + 0.1 * np.amax(v2)
    # variables, color_var = np.array([phi1, om1]), -np.exp(-cos(phi1 + phi2))
    # FxPath.see_path_1(1, variables, color_var, 'inferno', var_case=2, shift=(0., 0.), save="no")

    parameters[0] = r"Axe x : $x_2$"
    parameters[1] = r"Axe y : $y_2$"
    parameters[2] = r"Axe c : $v_2$"
    # variables, color_var = np.array([y2, x2]), v2
    # FxPath.see_path_1(lw, variables, color_var, color='inferno', var_case=1, save=savefig, displayedInfo=parameters)

    parameters[0] = r"Axe x : $\omega_1$"
    parameters[1] = r"Axe y : $\varphi_1 * \varphi_2$"
    parameters[2] = r"Axe c : $v_1 * v_2$"
    # variables, color_var = np.array([om1, phi1 * phi2]), v1 * v2
    # FxPath.see_path_1(lw, variables, color_var, color='inferno', var_case=2, save="no", displayedInfo=parameters)

    parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
    # FxPath.see_path(1, [np.array([phi1, om1]), np.array([phi2, om2])], [v2, v1],
    #                 ["inferno", "inferno"], [(0., 0.), (0., 0.)],
    #                 var_case=2, save="no", displayedInfo=parameters)

    see_animation(arrow_velocity=True, arrow_acceleration=True)

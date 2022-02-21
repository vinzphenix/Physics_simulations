import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from matplotlib.animation import FuncAnimation
from numpy import sin, cos, radians, pi
from scipy.integrate import odeint

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


def dynamics(u, _):
    du = np.zeros(4 * n_p)
    for i in range(n_p):
        phi1 = u[2 * i]
        phi2 = u[2 * i + 1]
        om1 = u[2 * (n_p + i)]
        om2 = u[2 * (n_p + i) + 1]
        Cos = cos(phi1 - phi2)
        Sin = sin(phi1 - phi2)
        l1 = long1[i]
        l2 = long2[i]
        du[2 * i] = om1
        du[2 * i + 1] = om2

        du[2 * n_p + 2 * i] = (m2 * Cos * (
                g * sin(phi2) - l1 * om1 * om1 * Sin) - m2 * l2 * om2 * om2 * Sin - M * g * sin(
            phi1) - D * l1 * om1) / (
                                  M * l1 - l1 * m2 * Cos ** 2)
        du[2 * n_p + 2 * i + 1] = 1 / (l2 * m1 * m2 * (-M + m2 * Cos ** 2)) * \
                                  (D * (l1 * m2 * om1 * Cos - l2 * m1 * om2) * (-M + m2 * Cos ** 2) - m1 * m2 * M * (
                                          g * cos(phi1) + l1 * om1 * om1) * Sin
                                   + m2 * m2 * (D * l1 * Sin * om1 - l2 * m1 * om2 * om2) * Sin * Cos)

    return du


def init():
    for line in lines:
        line.set_data([], [])
    time_text.set_text('')
    sector.set_theta1(90)
    return tuple(lines) + (time_text, sector)


def animate(i):
    i *= ratio
    for j, line in enumerate(lines):
        line.set_data([0, xa[i, j], xb[i, j]], [0, ya[i, j], yb[i, j]])
        # line.set_data([xb[i][2*j]], [yb[i][2*j]])

    time_text.set_text(time_template % (t[i + ratio - 1]))
    sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
    ax1.add_patch(sector)

    return tuple(lines) + (time_text, sector)


if __name__ == "__main__":

    #####     ================      Paramètres  de la simulation      ================      ####
    save = False
    n_p = 20  # [/]     -  nombre pendules

    g = 9.81  # [m/s²]  -  accélaration de pesanteur
    l1_avg = 1.0  # [m]     -  longueur 1er pendule
    l2_avg = 1.0  # [m]     -  longueur dernier pendule
    m1 = 1.0  # [kg]    -  masse 1er pendule
    m2 = 1.0  # [kg]    -  masse dernier pendule
    D = 0.0  # [kg/S]  -  coefficient frottement pendule

    # th = 0                 # [°]     -  angle 1er pendule
    # Om = 2 * sqrt(g / l1)  # [rad/s] -  v angulaire 1er pendule
    # phi = 0                # [°]     -  angle 2d pendule
    # om = 2 * sqrt(g / l2)  # [rad/s] -  v angulaire 2d pendule

    th, Om, phi, om = 150., 0., 150., 0.  # delta of 1/100 degree
    th, phi = radians([th, phi])

    # l1, l2, m1, m2 = 1., 1., 1., 1. / 19.
    # th, Om, phi, om = -0.68847079, -0.00158915, -0.9887163, 0.00234737
    # th, Om, phi, om = 0.000, 0.642 * sqrt(g/l1), -0.485, 0.796 * sqrt(g / l1)
    # th, Om, phi, om = 0.000, 0.642 * sqrt(g/l1), 2 * 0.485, 0.796 * sqrt(g / l1)

    Tend = 10.  # [s]    -  fin de la simulation
    n = int(600 * Tend)
    fps = 30
    ratio = n // (int(Tend * fps))

    ########################################################################################################

    #####     ================      Résolution de l'équation différentielle      ================      #####
    M = m1 + m2
    L = l1_avg + l2_avg
    long1 = np.linspace(l1_avg, l1_avg * 1.0, n_p)
    long2 = np.linspace(l2_avg, l2_avg * 1.0, n_p)

    U0 = np.zeros(4 * n_p)
    U0[0:2 * n_p:2] = np.linspace(th, th + pi / 180, n_p)
    U0[1:2 * n_p:2] = np.linspace(phi, phi + pi / 180, n_p)
    U0[2 * n_p::2] = np.linspace(Om, Om * 1.00, n_p)
    U0[2 * n_p + 1::2] = np.linspace(om, om * 1.00, n_p)

    tic = timer()
    t = np.linspace(0, Tend, n)
    sol = odeint(dynamics, U0, t)[:, :2 * n_p]
    print("      Elapsed time : %f seconds" % (timer() - tic))

    xa = np.multiply(sin(sol[:, 0::2]), long1)
    ya = np.multiply(-cos(sol[:, 0::2]), long1)
    xb = xa + np.multiply(sin(sol[:, 1::2]), long2)
    yb = ya + np.multiply(-cos(sol[:, 1::2]), long2)

    #####     ================      Création de la figure      ================      #####
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 6.5), constrained_layout=True)
    lim = 1.15 * L
    ax1.set_aspect("equal")
    ax1.axis([-lim, lim, -lim, lim])
    ax1.grid(ls=':')
    time_template = r'$t = %.2f s$'
    time_text = ax1.text(0.8, 0.93, '1', fontsize=15, wrap=True, transform=ax1.transAxes)
    cmap = plt.get_cmap('magma_r')  # magma,
    lines = []
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')
    for index in range(n_p):
        lineObj = ax1.plot([], [], 'o-', lw=1, color=cmap((index + 1) / n_p))[0]
        lines.append(lineObj)

    #####     ================      Animation      ================      #####
    # animate(0)
    # fig.savefig(f"./Animations/fig_quasi.png", format='png', bbox_inches='tight', dpi=300)
    anim = FuncAnimation(fig, animate, n // ratio, init_func=init, interval=5, blit=True, repeat_delay=3000)

    if save:
        # anim.save('Pendule_multiple_double_5.html', fps=30)
        anim.save('./Animations/{:s}.mp4'.format("multiple_pendulum_new"), writer="ffmpeg", dpi=150, fps=fps)
    else:
        plt.show()

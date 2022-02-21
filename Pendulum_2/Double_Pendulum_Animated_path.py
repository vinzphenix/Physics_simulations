import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy import sin, cos, hypot, sqrt, pi
from scipy.integrate import odeint
from timeit import default_timer as timer

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


def dynamics(u, _, l1, m1):
    th1, w1, th2, w2 = u
    Cos, Sin = cos(th1 - th2), sin(th1 - th2)
    M = m1 + m2

    f1 = (m2 * Cos * (g * sin(th2) - l1 * w1 * w1 * Sin) - m2 * l2 * w2 * w2 * Sin - M * g * sin(
        th1) - D * l1 * w1) / (M * l1 - l1 * m2 * Cos ** 2)
    f2 = 1. / (l2 * m1 * m2 * (-M + m2 * Cos ** 2)) * \
         (D * (l1 * m2 * w1 * Cos - l2 * m1 * w2) * (-M + m2 * Cos ** 2) - m1 * m2 * M * (
                 g * cos(th1) + l1 * w1 * w1) * Sin
          + m2 * m2 * (D * l1 * Sin * w1 - l2 * m1 * w2 * w2) * Sin * Cos)
    return np.array([w1, f1, w2, f2])


def init():
    line.set_segments([])
    mu_text.set_text(mu_template % (parameter[0]))
    return line, mu_text


def animate(frame):
    points = np.array([var_x[:, frame], var_y[:, frame]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line.set_segments(segments)
    line.set_array(colorarray[:, frame])
    mu_text.set_text(mu_template % (parameter[frame]))
    return line, mu_text


if __name__ == "__main__":

    #####     ================      Paramètres  de la simulation      ================      ####
    g = 9.81  # [m/sf]  -  accélaration de pesanteur
    D = 0.0  # [kg/s]  -  coefficient frottement

    # l1_avg = 1.00             # [m]     -  longueur 1er pendule
    # l2 = 1.00             # [m]     -  longueur 2em pendule
    # m1_avg = 12.0              # [kg]    -  masse 1er pendule
    # m2 = 1.0              # [kg]    -  masse 2em pendule
    #
    # phi1_0 = 40              # [°]     -  angle 1er pendule
    # om1_0 = 0                 # [rad/s] -  v angulaire 1er pendule
    # phi2_0 = 40*sqrt(2)       # [°]     -  angle 2e pendule
    # om2_0 = 0                 # [rad/s] -  v angulaire 2em pendule
    # phi1_0, phi2_0 = radians((phi1_0, phi2_0))

    phi1_0, phi2_0, om1_0, om2_0, l1_avg, l2, m1_avg, m2 = 2 / 9 * pi, 2 / 9 * pi * sqrt(2), 0, 0, 1.0, 1.0, 4.0, 1.0
    # phi1_0, phi2_0, om1_0, om2_0, l1_avg, l2, m1_avg, m2 = 0., 0., 3.5, -4 * sqrt(2), 1.00, 1.00, 1.00, 1.

    N = 50
    # LAB, MAB = np.linspace(0.75, 1.25, N), np.linspace(2., 2., N)
    L_array, M_array = np.linspace(0.75, 1.25, N), np.linspace(2., 2.2, N)

    Tend, fps = 50., 30
    n = int(100 * Tend)
    ratio = n // (int(Tend * fps))

    ########################################################################################################

    #####     ================      Résolution de l'équation différentielle      ================      #####

    L = L_array + l2, M_array + m2

    t = np.linspace(0, Tend, n)
    U0 = np.array([phi1_0, om1_0, phi2_0, om2_0])
    A = np.zeros((n, 4, N))

    tic = timer()
    for i in range(N):
        sol = odeint(dynamics, U0, t, args=(L_array[i], M_array[i]))
        A[:, :, i] = sol
    print("      Elapsed time : %f seconds" % (timer() - tic))

    #####     ================      Ecriture des Positions / Vitesses      ================      #####
    phi1, om1, phi2, om2 = A[:, 0, :], A[:, 1, :], A[:, 2, :], A[:, 3, :]
    x1, y1 = L_array * sin(phi1), -L_array * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
    vx2 = L_array * om1 * cos(phi1) + l2 * om2 * cos(phi2)
    vy2 = L_array * om1 * sin(phi1) + l2 * om2 * sin(phi2)
    v2 = hypot(vx2, vy2)

    lw, parameter = 1, M_array
    var_x, var_y, colorarray = x2, y2, v2
    bar, label = False, "speed of the pendulum [m/s]"
    save, save_name = False, 'Double_Pendulum_Animated_Path_1'

    #####     ================      Création de la figure      ================      #####
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    L_X, L_Y = np.amax(var_x) - np.amin(var_x), np.amax(var_y) - np.amin(var_y)

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(0, colorarray.max())
    line = LineCollection([], cmap=cmap, norm=norm, lw=lw)
    line.set_array(colorarray[:, 0])
    ax.add_collection(line)

    mu_template = r'$\mu = %7.4f$'
    mu_text = ax.text(0.02, 0.96, '', fontsize=15, transform=ax.transAxes)
    x_m, y_m = np.amin(var_x) - 0. * L_X, np.amin(var_y) - 0.2 * L_Y
    x_M, y_M = np.amax(var_x) + 0.2 * L_X, np.amax(var_y) + 0.2 * L_Y
    ax.axis([x_m, x_M, y_m, y_M])
    ax.set_aspect("equal")

    if bar:
        cbar = fig.colorbar(line)
        cbar.ax.set_ylabel(label)
        ax.grid(ls=":")
    else:
        plt.axis('off')

    #####     ================      Animation      ================      #####
    ani = FuncAnimation(fig, animate, N, interval=1, blit=True, init_func=init, repeat_delay=1000)
    if save:
        ani.save(save_name, fps=fps)
    else:
        plt.show()

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy import sin, cos, hypot, radians, linspace, array, amax, concatenate, amin, zeros
from scipy.integrate import odeint
from timeit import default_timer as timer

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # accélération de pesanteur
m = 1.0  # petite masse

r0 = 1  # position initiale  -  [m]
dr0 = 0.0  # vitesse initiale   -  [m/s]
th0 = 150  # angle initial      -  [°]
om0 = 0.0  # vitesse angulaire  -  [°/s]

Tend = 50
n = int(100 * Tend)
fps = 30
ratio = n // (int(Tend * fps))

N = 50

M1 = 4.501  # 6.014 # 4.419  # grosse masse
M2 = 4.700  # 6.114 # 6.014  # grosse masse
M_list = linspace(M1, M2, N)

th0a = 30
th0b = 150
th0_list = linspace(th0a, th0b, N)

# M_list = array(
#    [1.133, 1.172, 1.278, 1.337, 1.555, 1.655, 1.67, 1.904, 2.165, 2.165, 2.394, 2.8121, 2.945, 3.125, 3.52, 3.867,
#    4.1775, 4.745, 5.475, 5.68, 6.014, 6.806, 7.244, 7.49, 7.7, 8.182370012, 8.182370165])

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

t = linspace(0, Tend, n)
th0, om0 = radians((th0, om0))
U0 = array([r0, dr0, th0, om0])


def f(u, _, M):
    # r, dr, th, om = u
    f1 = (m * u[0] * u[3] * u[3] + g * (-M + m * cos(u[2]))) / (M + m)
    f2 = -1 / u[0] * (2 * u[1] * u[3] + g * sin(u[2]))
    return array([u[1], f1, u[3], f2])


A = zeros((n, 4, N))

tic = timer()
for i in range(N):
    sol = odeint(f, U0, t, args=(M_list[i],))
    # sol = odeint(f, array([r0, dr0, radians(th0_list[i]), om0]), t, args=(g, M_list[10], m))
    A[:, :, i] = sol
    if i % 10 == 0:
        print("Progression : {:d} / {:d}".format(i, N))
print("      Elapsed time : %f seconds" % (timer() - tic))

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

r, dr = A[:, 0, :], A[:, 1, :]
th, om = A[:, 2, :], A[:, 3, :]

x, y = r * sin(th), -r * cos(th)

x_min, x_max = amin(x), amax(x)
y_min, y_max = amin(y), amax(y)

vx = dr * sin(th) + r * om * cos(th)
vy = -dr * cos(th) + r * om * sin(th)
v = hypot(vx, vy)


########################################################################################################

#####     ================      Animation des Paths      ================      #####

def see_animation_path(lw, var_x, var_y, parameter, var_case=1, bar=False,
                       colorarray=v, color='jet', label="speed of the pendulum [m/s]",
                       save=None):
    #####     ================      Création de la figure      ================      #####

    # global ax
    plt.style.use('dark_background')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    L_X, L_Y = amax(var_x) - amin(var_x), amax(var_y) - amin(var_y)
    L_V = amax(colorarray[:, 0]) - amin(colorarray[:, 0])

    if var_case == 1:
        x_m, y_m = amin(var_x) - 0.2 * L_X, amin(var_y) - 0.2 * L_Y
        x_M, y_M = amax(var_x) + 0.2 * L_X, amax(var_y) + 0.2 * L_Y
        ax.axis([x_m, x_M, y_m, y_M])
        ax.set_aspect("equal", "datalim")
    elif var_case == 2:
        x_m, y_m = amin(var_x) - 0.4 * L_X, amin(var_y) - 0.05 * L_Y
        x_M, y_M = amax(var_x) + 0.4 * L_X, amax(var_y) + 0.05 * L_Y
        ax.axis([x_m, x_M, y_m, y_M])
    elif var_case == 3:
        ax.axis([-15, 2.5, -3.9, 3.9])
    elif var_case == 4:
        ax.axis([-20, 20, -4.25, 4.25])

    cmap = plt.get_cmap(color)
    norm = plt.Normalize(colorarray[:, 0].min() - L_V / 8, colorarray[:, 0].max())

    line = LineCollection([], cmap=cmap, norm=norm, lw=lw)
    line.set_array(colorarray[:, 0])
    ax.add_collection(line)

    if bar:
        cbar = fig.colorbar(line)
        ax.grid(ls=":")
        cbar.ax.set_ylabel(label)
        plt.subplots_adjust(left=0.05, right=0.90, bottom=0.08, top=0.92, wspace=None, hspace=None)
    else:
        plt.axis('off')
        plt.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=None, hspace=None)

    mu_template = r'$\mu = %.6f$'
    mu_text = ax.text(0.01, 0.96, '', fontsize=15, transform=ax.transAxes)
    patches.Wedge((x_max * 0.9, y_min), (x_max - x_min) / 20, theta1=90, theta2=90, color='lightgrey')

    ########################################################################################################

    #####     ================      Animation      ================      #####

    def init():
        line.set_segments([])
        mu_text.set_text(mu_template % (parameter[0]))
        return line, mu_text

    def animate(j):

        V_m, V_M = amin(colorarray[:, j]), amax(colorarray[:, j])
        points = array([var_x[:, j], var_y[:, j]]).T.reshape(-1, 1, 2)
        segments = concatenate([points[:-1], points[1:]], axis=1)
        line.set_segments(segments)
        line.set_norm(plt.Normalize(6 / 5 * V_m - V_M / 5, V_M))
        line.set_array(colorarray[:, j])

        mu_text.set_text(mu_template % (parameter[j]))

        return line, mu_text

    ani = FuncAnimation(fig, animate, N, interval=100,
                        blit=True, init_func=init,
                        repeat_delay=1000)

    if save is None:
        plt.show()
    else:
        ani.save(save, fps=20)


see_animation_path(1, r, dr, M_list, colorarray=r, color='inferno',
                   # var_case=4, bar=False, save="Atwood_om_dr_4.419_6.014.html")
                   var_case=2, bar=False, save=None)

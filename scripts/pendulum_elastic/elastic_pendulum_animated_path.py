import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy import sin, cos, hypot, radians, linspace, array, amax, concatenate, amin, zeros, mean, genfromtxt
from scipy.integrate import odeint
from timeit import default_timer as timer

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


#########################################################################################################

#####     ===========      Paramètres  de la simulation / Conditions initiales     ===========      #####

newRun = True
Tend = 30  # [s]    -  fin de la simulation

g = 9.81                  # accélération de pesanteur

l = 1.0  # longueur  - [m]
m = 1.00  # masse     - [kg]
k = 200  # coef de raideur - [N/m]

N = 100
mu = linspace(28, 34, N)

th0 = 10  # angle initial    -  [°]
om0 = 0  # vitesse initiale -  [°/s]
r0 = 0.875  # m * g / k
v0 = 0

n = int(50 * Tend)

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

t = linspace(0, Tend, n)
U0 = array([radians(th0), radians(om0), r0, v0])


def f(u, _, mu_):
    th_, om_, r_, v_ = u
    d2th_ = -2 * v_ * om_ / (r_ + l) - g * sin(th_) / (r_ + l)
    d2r_ = (r_ + l) * om_ * om_ + g * cos(th_) - mu_ * r_
    return array([om_, d2th_, v_, d2r_])


if newRun:
    A = zeros((n, 4, N))
    tic = timer()
    for i in range(N):
        sol = odeint(f, U0, t, args=(mu[i],))
        A[:, :, i] = sol
        if i % 10 == 0:
            print("Progression : {:d} / {:d}".format(i, N))
    print("      Elapsed time : %f seconds" % (timer() - tic))

    #savetxt("{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.txt".format(l, mu[0], mu[-1], th0, om0, r0, v0), A.reshape(n, 4 * N),  fmt="%.4e")

else:
    tic = timer()
    A = (genfromtxt("{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.txt".format(l, mu[0], mu[-1], th0, om0, r0, v0))
         ).reshape((n, 4, N))
    print("      Elapsed time : %f seconds" % (timer() - tic))

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

th, r = A[:, 0, :], A[:, 2, :]
om, v = A[:, 1, :], A[:, 3, :]

x =   (r + l) * sin(th)
y = - (r + l) * cos(th)

x_min, x_max = amin(x), amax(x)
y_min, y_max = amin(y), amax(y)

vx =   v * sin(th) + (r + l) * cos(th) * om
vy = - v * cos(th) + (r + l) * sin(th) * om
v  = hypot(vx, vy)


########################################################################################################

#####     ================      Animation des Paths      ================      #####

def see_animation_path(lw, var_x, var_y, parameter, var_case=1, bar=False,
                       colorarray=v, color='jet', label="speed of the pendulum [m/s]",
                       shift=0.0, saveVideo=None):

    #####     ================      Création de la figure      ================      #####

    plt.style.use('dark_background')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    L_X, L_Y = amax(var_x) - amin(var_x), amax(var_y) - amin(var_y)
    L, L_V = max(L_X, L_Y), amax(colorarray[:, 0]) - amin(colorarray[:, 0])

    if var_case == 1:
        x_m, y_m = mean(var_x) - 0.6 * L, mean(var_y) - 0.6 * L
        x_M, y_M = mean(var_x) + 0.6 * L, mean(var_y) + 0.6 * L
        ax.axis([x_m, x_M, y_m, y_M])
        ax.set_aspect("equal", "datalim")
        sector = patches.Wedge((0.90, 0.08), 0.04, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)
    elif var_case == 2:
        x_m, y_m = amin(var_x) - 0.6 * L_X, amin(var_y) - 0.05 * L_Y
        x_M, y_M = amax(var_x) + 0.6 * L_X, amax(var_y) + 0.05 * L_Y
        ax.axis([x_m, x_M, y_m, y_M])
        ax.set_aspect("equal", "datalim")
        rect = plt.Rectangle((0.1, 0.9), 0.1, 0.025, edgecolor='lightgrey', facecolor='black', transform=ax.transAxes)
        ax.add_patch(rect)
        sector = plt.Rectangle((0.1, 0.9), 0.1, 0.025, edgecolor='lightgrey',
                               facecolor='lightgrey', transform=ax.transAxes)
    elif var_case == 3:
        ax.axis([-0.375, 0.375, -0.98, -0.23])
        ax.set_aspect("equal", "datalim")
        sector = patches.Wedge((0.90, 0.08), 0.04, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)
    elif var_case == 4:
        ax.axis([-50, 50, -4.3, 4.3])

    cmap = plt.get_cmap(color)
    norm = plt.Normalize(colorarray[:, 0].min() - L_V * shift, colorarray[:, 0].max())

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

    mu_template = r'$\mu = %.1f$'
    mu_text = ax.text(0.85, 0.95, '', fontsize=15, transform=ax.transAxes)

    ax.text(0.51, 0.95, r'$l  = {:.2f} $'.format(l), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.51, 0.91, r'$r         = {:.2f} $'.format(r0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.51, 0.87, r'$v         = {:.2f} $'.format(v0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.68, 0.95, r'$\vartheta = {:.2f} $'.format(th0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.68, 0.91, r'$\omega    = {:.2f} $'.format(om0), fontsize=12, wrap=True, transform=ax.transAxes)

    ########################################################################################################

    #####     ================      Animation      ================      #####

    def init():
        line.set_segments([])
        mu_text.set_text(mu_template % (parameter[0]))
        if (var_case == 1) or (var_case == 3):
            sector.set_theta1(90)
        else:
            sector.set_width(0)
        return line, mu_text, sector

    def animate(j):
        V_m, V_M = amin(colorarray[:, j]), amax(colorarray[:, j])
        points = array([var_x[:, j], var_y[:, j]]).T.reshape(-1, 1, 2)
        segments = concatenate([points[:-1], points[1:]], axis=1)
        line.set_segments(segments)
        line.set_norm(plt.Normalize(6 / 5 * V_m - V_M / 5, V_M))
        line.set_array(colorarray[:, j])

        mu_text.set_text(mu_template % (parameter[j]))

        if (var_case == 1) or (var_case == 3):
            sector.set_theta1(90 - 360 * j / N)
        else:
            sector.set_width((j + 1) / N * 0.1)
        ax.add_patch(sector)

        return line, mu_text, sector

    anim = FuncAnimation(fig, animate, N, interval=350, blit=True, init_func=init, repeat_delay=1000)
    if saveVideo:
        anim.save('Pendule_Elastique_Video_1', fps=30)
    else:
        plt.show()


#see_animation_path(1, x, y, mu, colorarray=v, color='inferno', var_case=1, bar=False, saveVideo="XY_5_frame")
see_animation_path(1, x, y, mu, colorarray=v, color='inferno', var_case=2, bar=False, saveVideo=None)
#see_animation_path(1, om, r, mu, colorarray=v, color='inferno', var_case=2, bar=False, saveVideo="OM_R_1_frame")
#see_animation_path(1, om, v, mu, colorarray=v, color='inferno', var_case=2, bar=False, saveVideo="OM_V_1_frame")
#see_animation_path(1, r, v, mu, colorarray=v, color='inferno', var_case=2, bar=False, saveVideo="TH_OM_1_frame")

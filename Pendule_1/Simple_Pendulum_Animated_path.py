import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, hypot, radians, linspace, pi, degrees, zeros
from scipy.integrate import odeint
from timeit import default_timer as timer

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81              # [m/sf]  -  accélaration de pesanteur
l = 1.00             # [m]     -  longueur 1er pendule
m = 12.0              # [kg]    -  masse 1er pendule
D  = 0.0              # [kg/s]  -  coefficient de frottement

phi_0 = 40              # [°]     -  angle 1er pendule
om_0 = 0                 # [rad/s] -  v angulaire 1er pendule

Tend = 40
n = int(1050 * Tend)
fps = 30
ratio = n // (int(Tend * fps))

N = 100
#LA = 0.75 ; LB = 1.25
#LAB = linspace(LA, LB, N)
#MA = 3.75 ; MB = 4.25
#MAB =  linspace(MA, MB, N)

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

t = linspace(0, Tend, n)
U0 = zeros((N, 2))
U0[:, 0] = radians(linspace(170, 179.9, N))
U0[:, 1] = zeros(N)

f = lambda u, _: np.array([u[1], - g / l * sin(u[0]) - D / m * u[1]])
A = zeros((n, 2, N))

tic = timer()
for i in range(N):
    sol = odeint(f, U0[i], t)
    A[:, :, i] = sol
print("      Elapsed time : %f seconds" % (timer() - tic))

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

phi = A[:, 0, :]
om = A[:, 1, :]

x, vx = l * sin(phi), l * om * cos(phi)
y, vy = -l * (cos(phi)), l * om * sin(phi)
v = hypot(vx, vy)


########################################################################################################

#####     ================      Animation des Paths      ================      #####

def see_animation_path(var_x, var_y, parameter):

    #####     ================      Création de la figure      ================      #####

    #plt.style.use('dark_background')

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), constrained_layout=True)
    ax.axis([-Tend * 0.1, Tend * 1.1, -pi * 1.1, pi * 1.1])
    ax.grid(ls=":")

    line1, = ax.plot([], [], '-', lw=1, color='C1')

    mu_template = r'$\theta_0 = %.3f$'
    mu_text = ax.text(0.02, 0.96, '', fontsize=15, transform=ax.transAxes)

    ########################################################################################################

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        mu_text.set_text(mu_template % (parameter[0]))
        return line1, mu_text

    def animate(idx):
        line1.set_data([var_x], [var_y[:, idx]])
        mu_text.set_text(mu_template % (degrees(parameter[idx])))
        return line1, mu_text

    _ = FuncAnimation(fig, animate, N, interval=20, blit=True, init_func=init, repeat_delay=5000)
    plt.show()


see_animation_path(t, phi, U0[:, 0])

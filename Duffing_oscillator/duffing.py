import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from scipy.integrate import odeint


########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

alpha, beta = 1.0, 5.0

gamma = 8
delta = 0.02
w = 0.5

dt = 0.05  # [s]    -  pas de temps
Tend = 200  # [s]    -  fin de la simulation

fps = int(1 / dt)  # animation en temps réel

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
n = int(Tend // dt) + 1
t = np.linspace(0, Tend, n + 1)


def f(u_, t_):
    u1, u2 = u_
    f1 = - alpha * u1 - beta * u1 ** 3 - delta * u2 + gamma * cos(w * t_)
    return np.array([u2, f1])


points_x = []
points_y = []

for iteration in range(50):
    x0, dx0 = 1. + rd.rand(), 4 * (rd.rand() - 0.5)
    U0 = np.array([x0, dx0])

    sol = odeint(f, U0, t)
    x, dx = sol.T

    for i in range(1, n):
        u_prev, v_prev = x[i-1], dx[i-1]
        u, v = x[i], dx[i]
        t_pr, t_nx = t[i-1], t[i]

        if (sin(t_pr*w) * sin(t_nx*w) <= 0.) and (cos(t_pr*w) > 0.):
            t_pr, t_nx = np.remainder(np.array([t_pr*w, t_nx*w]) + pi, 2 * pi) - pi
            coef = t_pr / (t_pr - t_nx)

            points_x.append(u_prev * (1 - coef) + u * coef)
            points_y.append(v_prev * (1 - coef) + v * coef)


########################################################################################################

#####     ================      Création de la figure      ================      #####

fig, ax = plt.subplots(1, 1, figsize=(9, 6), constrained_layout=True)
ax.plot(points_x, points_y, 'o', markersize=0.5, color='black')
plt.show()

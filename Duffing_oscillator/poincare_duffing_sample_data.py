import matplotlib
import numpy as np

from numpy import random as rd
from numpy import cos, pi, array, zeros, fmod

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def runge(n, dt, U0, nSteps):
    def f(u, tau):
        u1, u2 = u
        f1 = - alpha * u1 - beta * u1 ** 3 - gamma * u2 + A * cos(w * tau)
        return array([u2, f1])

    point_list = zeros((n, nPoints, 2))
    index = 0
    k = 0

    for i in range(nSteps - 1):

        t = dt * i

        K1 = f(U0, t)
        K2 = f(U0 + K1 * dt / 2, t)
        K3 = f(U0 + K2 * dt / 2, t)
        K4 = f(U0 + K3 * dt, t)
        U1 = U0 + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

        x_1, v_1 = U0
        x_2, v_2 = U1
        t_1, t_2 = t - dt, t
        t_1, t_2 = fmod(array([t_1 * w, t_2 * w]) - 2 * pi * k / n, 2 * pi)

        if (t_2 < t_1) and (240. < t):

            coef = t_1 / (t_1 - t_2)
            point_x = x_1 * (1 - coef) + x_2 * coef
            point_y = v_1 * (1 - coef) + v_2 * coef

            point_list[k, index] = point_x, point_y
            k += 1

            if k == n:
                index += 1
                k = 0
                if index == nPoints:
                    return index, point_list

        U0 = U1

    print("Not enough", index, nPoints, nSteps)
    return index, point_list


if __name__ == "__main__":
    alpha, beta, gamma, A, w = 1.0, 5.0, 0.02, 8.0, 0.5
    # alpha, beta, gamma, A, w = -1.0, 1.0, 0.15, 0.3, 1.0
    # alpha, beta, gamma, A, w = -1.0, 1.0, 0.1, 0.35, 1.4

    nFigs = 400
    nRuns = 10
    nPoints = 135

    nMax = 500000
    delta_t = 0.005

    x_lim = (-1.5, 1.5)
    y_lim = (-0.5, 1.0)
    L_x, L_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    m_x, m_y = (x_lim[1] + x_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2

    filename = "samples_2.txt"
    file = open(filename, "a")

    file.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(alpha, beta, gamma, A, w, delta_t))
    file.write("{:d} {:d} {:d}\n".format(nFigs, nRuns, nPoints))

    for run in range(nRuns):
        print("itération {:>2d}".format(run + 1), end="")
        x0, dx0 = L_x * (rd.rand() - 0.5) + m_x, L_y * (rd.rand() - 0.5) + m_y
        U_init = np.array([x0, dx0])
        nbPoints, points = runge(nFigs, delta_t, U_init.copy(), nMax)

        for j in range(nFigs):
            for k_ in range(nbPoints):
                file.write("{:d} {:.5f} {:.5f}\n".format(j, points[j, k_, 0], points[j, k_, 1]))

        print(" : ok")

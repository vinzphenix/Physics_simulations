import matplotlib.pyplot as plt
import numpy as np

from scipy.special import jv, jn_zeros
from scipy.integrate import quad, dblquad
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from numpy import sqrt, sin, cos, linspace, pi, outer, exp, ones, zeros, newaxis, repeat, abs

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 15, 13


# initial conditions f
f1 = lambda r_, t_: 10 * (
        (1 - (r_ / 2) ** 2) * sin(r_ * cos(t_) / 2) + 0.4 * (1 - (r_ / 2)) * sin(2 * r_ * sin(t_)) ** 2) ** 2  # A
f2 = lambda r_, t_: 1 / 2 * (2 - sqrt((4 * (r_ * cos(t_) - 0.5) ** 2) + 4 * (r_ * sin(t_)) ** 2)) * (2 - r_)  # B
f3 = lambda r_, t_: 3 / 4 * (cos(pi / b * (2 * 1 + 1) / 2 * r_)) * cos(r_ ** 2 * cos(t_) * sin(t_))  # C
f4 = lambda r_, t_: 4 / 5 * (cos(pi / b * (2 * 0 + 1) / 2 * r_)) * (1 - r_ * cos(t_)) * (1 - r_ * sin(t_))  # D
f5 = lambda r_, t_: 1 / 1 * (cos(pi / b * (2 * 0 + 1) / 2 * r_)) * (1 - r_ * cos(t_)) * (
        0.5 - r_ ** 2 * cos(t_) * sin(t_))  # E


def update_plot(idx):
    ax.view_init(elev=22 - 6 * idx / nt, azim=120 + 30 * idx / nt)
    time_text.set_text(time_template.format(t[idx]))
    plot[0].remove()

    plot[0] = ax.plot_surface(x, y, U[:, :, idx], linewidth=0, antialiased=False, rstride=1, cstride=1,
                              facecolors=cmap(sqrt((abs(U[:, :, idx]) / np.amax(U)))))


if __name__ == "__main__":

    f = f2  # initial condition

    c = 1  # speed of the wave
    h = 0.02  # damping coefficient
    b = 2  # radius of the membrane
    #Tend = 10 * 2 * pi / (sqrt((jn_zeros(2, 2)[-1]) ** 2 + h ** 2))  # duration of the simulation
    Tend = 8.
    fps = 20  # frames per second

    nr = 50  # 100  # number of steps in the r direction
    nth = 50  # 100  # number of steps in the theta direction
    nt =  int(fps * Tend) + 1  # number of time steps

    n = 8  # number of terms in the summation
    m = 5

    r = linspace(0, b, nr)
    th = linspace(0, 2 * pi, nth)
    t = linspace(0, Tend, nt)

    k = zeros((m, n))  # normal modes
    A = zeros((m, n))  # coefficients of the summation
    B = zeros((m, n))
    C = zeros((m, n))

    U = zeros((nr, nth, nt))

    for i in range(m):
        k[i, :] = jn_zeros(i, n) / b
        for j in range(n):
            # print("m = {:.0f} \t n = {:.0f}".format(i, j))
            kmn = k[i, j]

            if i == 0:
                g1 = lambda r_, th_: jv(i, kmn * r_) * f(r_, th_) * r_
                g2 = lambda r_: (jv(i, kmn * r_)) ** 2 * r_
                I = dblquad(g1, 0, 2 * pi, 0, b)[0] / (2 * pi * quad(g2, 0, b)[0])
                C[i, j] = 1.
                TH = ones(nth)

            else:
                g1c = lambda r_, th_: jv(i, kmn * r_) * f(r_, th_) * r_ * cos(i * th_)
                g1s = lambda r_, th_: jv(i, kmn * r_) * f(r_, th_) * r_ * sin(i * th_)
                g2 = lambda r_: (jv(i, kmn * r_)) ** 2 * r_

                Ic = dblquad(g1c, 0, 2 * pi, 0, b)[0] / (pi * quad(g2, 0, b)[0])
                Is = dblquad(g1s, 0, 2 * pi, 0, b)[0] / (pi * quad(g2, 0, b)[0])
                if Is == 0.0:
                    print(f"m={m} n={n} : 'Is' = 0")
                    continue

                # print("Ic = {:.2e} \t Is = {:.2e}".format(Ic, Is))
                C[i, j] = Ic / Is
                TH = (C[i, j]) * cos(i * th) + sin(i * th)
                I = Is

            if c * c * kmn * kmn < h * h:
                alpha1 = -h + sqrt(h * h - c * c * kmn * kmn)
                alpha2 = -h - sqrt(h * h - c * c * kmn * kmn)

                A[i, j] = I * alpha2 / (alpha2 - alpha1)
                B[i, j] = I * alpha1 / (alpha1 - alpha2)
                T = A[i, j] * exp(alpha1 * t) + B[i, j] * exp(alpha2 * t)

            if c * c * kmn * kmn > h * h:
                beta = sqrt(c * c * kmn * kmn - h * h)
                A[i, j] = I
                B[i, j] = I * h / beta
                T = exp(-h * t) * (A[i, j] * cos(beta * t) + B[i, j] * sin(beta * t))

            else:
                A[i, j] = I
                B[i, j] = I * h
                T = exp(-h * t) * (A[i, j] + B[i, j] * t)

            R = jv(i, kmn * r)

            # print("A = {:.2e} \t B = {:.2e} \t C = {:.2e}".format(A[i, j], B[i, j], C[i, j]))
            TT = repeat(T[newaxis, :], nth, axis=0)
            TTT = repeat(TT[newaxis, :, :], nr, axis=0)
            U += repeat(outer(R, TH)[:, :, newaxis], nt, axis=2) * TTT

    x = outer(r, cos(th))
    y = outer(r, sin(th))

    # print("U : {} \n E : {} \n T : {} \n R : {}".format(U.shape, E.shape, T.shape, R.shape))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    # ax = Axes3D(fig, proj_type='persp')
    cmap = plt.get_cmap('jet')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # delete grey background and the numbers on the axis
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    plot = [ax.plot_surface(x, y, U[:, :, 0], cmap=cmap, rstride=1, cstride=1)]

    time_template = r'$t = {:.2f} \; s$'
    time_text = ax.text2D(0.8, 0.9, '', fontsize=ftSz2, transform=ax.transAxes)

    ax.set_xlim(-b * 1.05, b * 1.05)
    ax.set_ylim(-b * 1.05, b * 1.05)
    ax.set_zlim(-b * 1.05, b * 1.05)

    anim = FuncAnimation(fig, update_plot, nt)
    save = ""

    if save == "gif":
        writerGIF = PillowWriter(fps=fps)
        anim.save(f"./anim_circular.gif", writer=writerGIF, dpi=100)
    elif save == "html":
        anim.save('circular_membrane_II_B.html', fps=fps)
    else:
        plt.show()


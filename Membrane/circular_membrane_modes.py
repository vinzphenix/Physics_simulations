import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv, jn_zeros
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from numpy import sqrt, sin, cos, linspace, pi, outer, exp, newaxis, repeat, abs


def update_plot(idx):
    ax.view_init(elev=22 - 4 * idx / nt, azim=120 + 30 * idx / nt)
    time_text.set_text(time_template % (t[idx]))
    plot[0].remove()

    plot[0] = ax.plot_surface(x, y, U[:, :, idx], linewidth=0, antialiased=False, rstride=1, cstride=1,
                              facecolors=cmap(sqrt((abs(U[:, :, idx]) / np.amax(U)))))


if __name__ == "__main__":
    c = 1  # speed of the wave
    h = 0.2  # damping coefficient
    b = 1  # radius of the membrane
    #Tend = 10 * 2 * pi / (sqrt((jn_zeros(2, 2)[-1]) ** 2 + h ** 2))  # duration of the simulation
    Tend = 3
    fps = 30  # frames per second

    nr = 40  # number of steps in the r direction
    nth = 40  # number of steps in the theta direction
    nt =  int(fps * Tend)  # number of time steps

    n = 2  # radial mode index
    m = 1  # angular mode index

    r = linspace(0, b, nr)
    th = linspace(0, 2 * pi, nth)
    t = linspace(0, Tend, nt)

    k = jn_zeros(m, n + 1) / b
    kmn = k[n]
    TH = cos(m * th)

    if c * c * kmn * kmn < h * h:
        alpha1 = -h + sqrt(h * h - c * c * kmn * kmn)
        alpha2 = -h - sqrt(h * h - c * c * kmn * kmn)
        A = alpha2 / (alpha2 - alpha1)
        B = alpha1 / (alpha1 - alpha2)
        T = A * exp(alpha1 * t) + B * exp(alpha2 * t)
    elif c * c * kmn * kmn > h * h:
        beta = sqrt(c * c * kmn * kmn - h * h)
        A, B = 1, h / beta
        T = exp(-h * t) * (A * cos(beta * t) + B * sin(beta * t))
    else:
        A, B = 1, h
        T = exp(-h * t) * (A + B * t)

    R = jv(m, kmn * r)
    T = repeat(T[newaxis, :], nth, axis=0)
    T = repeat(T[newaxis, :, :], nr, axis=0)
    U = repeat(outer(R, TH)[:, :, newaxis], nt, axis=2) * T

    x = outer(r, cos(th))
    y = outer(r, sin(th))

    # print("U : {} \n E : {} \n T : {} \n R : {}".format(U.shape, E.shape, T.shape, R.shape))

    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig, proj_type='persp')
    cmap = cm.get_cmap('jet')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # delete grey background and the numbers on the axis
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    plot = [ax.plot_surface(x, y, U[:, :, 0], cmap=cmap, rstride=1, cstride=1)]

    time_template = r'$t = %.1fs$'
    time_text = ax.text2D(0.8, 0.9, '', fontsize=12, transform=ax.transAxes)

    ax.set_xlim(-b * 1.05, b * 1.05)
    ax.set_ylim(-b * 1.05, b * 1.05)
    ax.set_zlim(-b * 1.05, b * 1.05)

    animate = FuncAnimation(fig, update_plot, nt, interval=20)
    # animate.save(f"circular_membrane_mode_{m:d}_{n:d}.html", fps=fps)
    plt.show()

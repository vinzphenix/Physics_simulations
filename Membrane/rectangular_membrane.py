import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from numpy import sqrt, sin, cos, linspace, pi, outer, exp, zeros, newaxis, repeat, hypot, meshgrid, abs

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 15, 13


# initial condition f
f1 = lambda xi, eta: 0.03 * xi * (L - xi) * eta * (H - eta) * exp(xi * eta)  # 2
f2 = lambda xi, eta: 0.15 * xi * (L - xi) ** 3 * eta * (H - eta) ** 2  # 3
f3 = lambda xi, eta: (1 + 0.2 * sin(7 * sqrt(L) * xi * eta)) * 5 * xi / L ** 2 * (xi - L) * eta * (eta - H)  # 4
f4 = lambda xi, eta: 2 * (sin(pi * xi / L) * sin(pi * eta / H) + 0.08 * sin(7 * pi * xi / L) * sin(9 * pi * eta / H)) \
                     * (L - xi) * exp(-0.5 * (xi - L) ** 2)  # 5
f5 = lambda xi, eta: 2.5 * (exp(-3 * (xi - L / 4) ** 2 - 3 * (eta - H / 4) ** 2) - exp(
    - 3 * (xi - 3 * L / 4) ** 2 - 3 * (eta - 3 * H / 4) ** 2)) * sin(pi * xi / L) * sin(pi * eta / H)  # 6-7


def update_plot(idx):
    ax.view_init(elev=20 - 6 * idx / nt, azim=20 + 30 * idx / nt)
    time_text.set_text(time_template.format(t[idx]))

    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, U[:, :, idx], facecolors=cmap(sqrt(1 / Umax * abs(U[:, :, idx]))))


if __name__ == "__main__":

    f = f5  # initial condition

    c = 1  # speed of the wave
    h = 0.01  # damping coefficient
    L = 3.    # length of rectangle
    H = 2.    # height of rectangle
    Tend = 10.  # duration of the simulation
    fps = 15  # frames per second

    nx = 90  # number of steps in the r direction
    ny = 60  # number of steps in the theta direction
    nt = int(fps * Tend)+1  # number of time steps
    n = 15  # number of terms in the n summation
    m = 15  # number of terms in the m summation

    x = linspace(0, L, nx)
    y = linspace(0, H, ny)
    t = linspace(0, Tend, nt)

    A = zeros((n, m))  # coefficients of the summation
    B = zeros((n, m))  # coefficients of the summation
    U = zeros((nx, ny, nt))

    for i in range(n):
        for j in range(m):
            pn = (i+1) * pi / L
            qm = (j+1) * pi / H
            knm = hypot(pn, qm)

            g = lambda eta, xi: f(xi, eta) * sin(pn *  xi) * sin(qm * eta)

            I = 4 / (L * H) * dblquad(g, 0, L, lambda xi: 0, lambda xi: H)[0]
            #print(i+1, "\t", j + 1, "\t", I)

            X = sin(pn*x)
            Y = sin(qm*y)

            if c * c * knm * knm < h * h:
                alpha1 = -h + sqrt(h * h - c * c * knm * knm)
                alpha2 = -h - sqrt(h * h - c * c * knm * knm)
                A[i, j] = I * alpha2 / (alpha2 - alpha1)
                B[i, j] = I * alpha1 / (alpha1 - alpha2)
                T = A[i, j] * exp(alpha1 * t) + B[i, j] * exp(alpha2 * t)

            elif c * c * knm * knm > h * h:
                beta = sqrt(c * c * knm * knm - h * h)
                A[i, j] = I
                B[i, j] = I * h / beta
                T = exp(-h * t) * (A[i, j] * cos(beta * t) + B[i, j] * sin(beta * t))

            else:
                A[i, j], B[i, j] = I, I * h
                T = exp(-h * t) * (A + B * t)

            TT = repeat(T[newaxis, :], ny, axis=0)
            TTT = repeat(TT[newaxis, :, :], nx, axis=0)
            U += repeat(outer(X, Y)[:, :, newaxis], nt, axis=2) * TTT

    x, y = meshgrid(x, y)
    x = x.T
    y = y.T

    # print("U : {} \n E : {} \n T : {} \n R : {}".format(U.shape, E.shape, T.shape, R.shape))

    fig = plt.figure(figsize=(8, 8), constrained_layout=False)
    ax = fig.add_subplot(projection='3d')
    # ax = Axes3D(fig, proj_type='persp')
    cmap = plt.get_cmap('jet')

    ax.set_xlim(0, max(L, H))
    ax.set_ylim(0, max(L, H))
    ax.set_zlim(-0.5 * max(L, H), 0.5 * max(L, H))

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # delete grey background and the numbers on the axis
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    plot = [ax.plot_surface(x, y, U[:, :, 0], color='0.75', rstride=1, cstride=1)]

    time_template = r'$t = {:.2f} \; s$'
    time_text = ax.text2D(0.8, 0.9, '', fontsize=ftSz2, transform=ax.transAxes)
    Umax = abs(U).max()

    anim = FuncAnimation(fig, update_plot, nt, interval=20, repeat_delay=3000, repeat=True)
    save = ""

    if save == "gif":
        writerGIF = PillowWriter(fps=fps)
        anim.save(f"./anim_rectangular.gif", writer=writerGIF, dpi=100)
    elif save == "html":
        anim.save('rectangular_membrane_7.html', fps=fps)
    else:
        plt.show()

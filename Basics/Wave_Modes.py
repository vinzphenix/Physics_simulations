import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import sqrt, sin, pi
from matplotlib.animation import FuncAnimation

ftSz1, ftSz2 = 17, 13
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
matplotlib.rcParams['axes.labelsize'] = 15


nx, nt = 600, 300 * 2
L, F, mu, Tend = 8., 0.5, 0.25, 20.
c = sqrt(F / mu)

# l1, l2 = 0.27, 0.26
# k1, k2 = 2 * pi / np.array([l1, l2])
# w1, w2 = c * np.array([k1, k2])

x = np.linspace(0, L, nx)
t = np.linspace(0, Tend, nt)

U = np.zeros((nt, nx))
i_list = [1, 3, 4, 7, 8]
A = np.array([1, 0.1, -0.15, 0.1, -0.05])

for idx, i in enumerate(i_list):
    T = sin(c * (i * pi / L) * t)
    X = sin((i * pi / L) * x)
    U += A[idx] * np.outer(T, X)

fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
# ax = fig.add_subplot(111, xlim=(0, L), ylim=(1.5 * U.min(), 1.5 * U.max()), aspect='equal',
#                      xlabel=r'$x \: [m] $', ylabel=r'$\xi(x) \: [m]$')
ax.grid(ls=':')
ax.set_xlabel(r"$x \: [m] $")
ax.set_ylabel(r"$\xi(x) \: [m]$")
ax.set_aspect("equal", "datalim")
ax.axis([0, L, 1.5 * np.amin(U), 1.5 * np.amax(U)])

ax.set_title(r'$\xi(x,t) = \sum_{n=1}^{\infty} \: A_n \cdot \sin{(k_n x)} \cdot \sin{(\omega_n t)} $',
             fontsize=ftSz1)

positions = np.linspace(0.185, 0.75, len(i_list))
for i, a, pos in zip(i_list, A, positions):
    ax.text(pos, 0.94, r'$A_{{{:d}}} = {:.2f}$'.format(i, a), wrap=True, transform=ax.transAxes, fontsize=ftSz2)

line, = ax.plot([], [], '-', lw=2, color='C0')
time_template = r'$t = %.2f$'
time_text = ax.text(0.02, 0.94, '', fontsize=14, wrap=True, transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animated(frame):
    line.set_data([x, U[frame]])
    time_text.set_text(time_template % (t[frame]))
    return line, time_text


ani = FuncAnimation(fig, animated, nt, interval=1,
                    blit=True, init_func=init, repeat_delay=1000)
# ani.save('Wave_Modes_3.html', fps=30)
plt.show()

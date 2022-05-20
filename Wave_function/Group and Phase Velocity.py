import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from matplotlib.animation import FuncAnimation

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['axes.labelsize'] = 15

Tend, Xend = 23, 3
nx, nt = 600, Tend * 30
c = 0.25

n1, l1 = 1.46, 0.150
n2, l2 = 1.60, 0.125
k1, k2 = 2 * pi / np.array([l1, l2])
w1, w2 = np.array([k1, k2]) * c / np.array([n1, n2])

vp = (w2 + w1) / (k2 + k1)
vg = (w2 - w1) / (k2 - k1)

x = np.linspace(0, Xend, nx)
t = np.linspace(0, Tend, nt)

X, T = np.meshgrid(x, t)

U1 = sin(k1 * X - w1 * T)
U2 = sin(k2 * X - w2 * T)
U = U1 + U2
U_e = 2 * cos((k2 - k1) / 2 * X - (w2 - w1) / 2 * T)
u_ph = sin(k1 * (pi / (k1 + k2) + vp * t) - w1 * t) + sin(k2 * (pi / (k1 + k2) + vp * t) - w2 * t)

fig, axs = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
# fig.suptitle(r'$Phase \: and \: Group \: Velocity$')
fig.suptitle(r'$Phase \: and \: Group \: Velocity$' + '\n'
             + r'$c = %.2f ; \; \; v_1 = %.2f ; \; \; v_2 = %.2f ; \; \; v_{phase} = %.3f ; \;'
               r' \; v_{groupe} = %.2f $' % (c, c / n1, c / n2, vp, vg), fontsize=18)

axs[1].set_ylabel(r"$x \: [m] $")
for ax in axs:
    ax.axis([0, Xend, 1.1 * U.min(), 1.1 * U.max()])
    ax.set_ylabel(r"$E(x) \: [V/m]$")
    ax.grid(ls=':')

line1, = axs[0].plot([], [], '-', lw=2, color='C0')
line2a, = axs[0].plot([], [], ':', lw=1, color='grey')
line2b, = axs[0].plot([], [], ':', lw=1, color='grey')
line3, = axs[1].plot([], [], '-', lw=1, color='C1')
line4, = axs[1].plot([], [], '-', lw=1, color='C2')

mark_ph1, = axs[1].plot([], [], marker='o', ms=8, color='C1')
mark_ph2, = axs[1].plot([], [], marker='o', ms=8, color='C2')
mark_ph, = axs[0].plot([], [], marker='o', ms=8, color='C0')
mark_gr, = axs[0].plot([], [], marker='o', ms=8, color='grey')

time_template = r'$t = %.2f s$'
time_text = axs[0].text(0.02, 0.93, '', fontsize=14, wrap=True, transform=axs[0].transAxes)


def init():
    line1.set_data([], [])
    line2a.set_data([], [])
    line2b.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])

    mark_ph.set_data([], [])
    mark_gr.set_data([], [])
    mark_ph1.set_data([], [])
    mark_ph2.set_data([], [])

    time_text.set_text('')

    return line1, line2a, line2b, line3, line4, mark_ph, mark_gr, mark_ph1, mark_ph2, time_text


def animated(i):
    line1.set_data([x, U[i]])
    line2a.set_data([x, U_e[i]])
    line2b.set_data([x, -U_e[i]])
    line3.set_data([x, U1[i]])
    line4.set_data([x, U2[i]])

    mark_ph.set_data(pi / (k1 + k2) + vp * t[i], u_ph[i])
    mark_gr.set_data(vg * t[i], 2)
    mark_ph1.set_data(l1 / 4 + c / n1 * t[i], 1)
    mark_ph2.set_data(l2 / 4 + c / n2 * t[i], 1)

    time_text.set_text(time_template % (t[i]))
    return line1, line2a, line2b, line3, line4, mark_ph, mark_gr, mark_ph1, mark_ph2, time_text


anim = FuncAnimation(fig, animated, nt, interval=1, blit=True, init_func=init, repeat_delay=1000)
# anim.save('Group_Velocity_2.html', fps=30)
plt.show()

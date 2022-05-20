import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from matplotlib.animation import FuncAnimation, PillowWriter

save = ""
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = (save == "gif") or (save == "snapshot")
ftSz1, ftSz2, ftSz3 = 20, 16, 14


c = 0.25

# n1, n2, l1, l2  = 1.46, 1.60, 0.150, 0.125  # Positive
# n1, n2, l1, l2 = 1.50, 1.50*1.2, 0.150, 0.150/1.2  # Zero
n1, n2, l1, l2 = 1.50, 1.50*1.3, 0.150, 0.150 / 1.15

k1, k2 = 2 * pi / np.array([l1, l2])
w1, w2 = np.array([k1, k2]) * c / np.array([n1, n2])

vp = (w2 + w1) / (k2 + k1)
vg = (w2 - w1) / (k2 - k1)

Tend, Xend = np.abs(2 * pi / (w2 - w1)), 3
# Tend, Xend = 8., 3
fps = 7.5
nx, nt = 600, int(Tend * fps)

x = np.linspace(0, Xend, nx)
t = np.linspace(0, Tend, nt)

X, T = np.meshgrid(x, t)

U1 = sin(k1 * X - w1 * T)
U2 = sin(k2 * X - w2 * T)
U = U1 + U2
U_e = 2 * cos((k2 - k1) / 2 * X - (w2 - w1) / 2 * T)
u_ph = sin(k1 * (pi / (k1 + k2) + vp * t) - w1 * t) + sin(k2 * (pi / (k1 + k2) + vp * t) - w2 * t)

fig, axs = plt.subplots(2, 1, figsize=(11., 6.), sharex="all")

# fig.suptitle(r'$Phase \: and \: Group \: Velocity$')
fig.suptitle(r'$\rm {{Phase \: and \: Group \: Velocity}}$' '\n' '$c = {:.2f} \qquad '
            'v_1 = {:.2f} \qquad v_2 = {:.2f} \qquad v_{{phase}} = {:.3f} \qquad '
            'v_{{group}} = {:.3f} $'.format(c, c / n1, c / n2, vp, vg), fontsize=ftSz2)

axs[1].set_xlabel(r"$x \; [m] $", fontsize=ftSz2)
for ax in axs:
    ax.axis([0, Xend, 1.1 * U.min(), 1.1 * U.max()])
    ax.set_ylabel(r"$E(x) \;\; \rm[V\,/\,m]$", fontsize=ftSz2)
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

time_template = r'$t = {:.2f} \; s$'
time_text = axs[1].text(0.02, 0.84, '', fontsize=ftSz2, wrap=True, transform=axs[1].transAxes)
fig.tight_layout()


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


def update(i):
    line1.set_data([x, U[i]])
    line2a.set_data([x, U_e[i]])
    line2b.set_data([x, -U_e[i]])
    line3.set_data([x, U1[i]])
    line4.set_data([x, U2[i]])

    mark_ph.set_data(pi / (k1 + k2) + vp * t[i], u_ph[i])
    mark_gr.set_data(vg * t[i], 2)
    mark_ph1.set_data(l1 / 4 + c / n1 * t[i], 1)
    mark_ph2.set_data(l2 / 4 + c / n2 * t[i], 1)

    time_text.set_text(time_template.format(t[i]))
    return line1, line2a, line2b, line3, line4, mark_ph, mark_gr, mark_ph1, mark_ph2, time_text


anim = FuncAnimation(fig, update, nt, interval=1, blit=True, init_func=init, repeat_delay=1000)

if save == "html":
    anim.save('group_velocity_2.html', fps=30)
elif save == "gif":
    anim.save('./group_velocity.gif', writer=PillowWriter(fps=20), dpi=100)
elif save == "snapshot":
    update(int(12. * nt / Tend))
    fig.savefig("./group_velocity.svg", format="svg", bbox_inches="tight")
else:
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

nx, nt = 150, 300
L, c = 1., 2.
phi0 = 1
x_array = np.linspace(-2 * L, 3 * L, nx)
t_array = np.linspace(0, L, nt)
F = np.zeros((nt, nx))


def phi(x, t):
    def f(s):
        if 0 < s <= L / 2:
            return phi0 * 2 * (s / L)
        elif L / 2 < s <= L:
            return phi0 * 2 * (1 - s / L)
        else:
            return 0

    return 0.5 / phi0 * (f(x - c * t) + f(x + c * t))


for i in range(nt):
    for j in range(nx):
        F[i, j] = phi(x_array[j], t_array[i])

fig = plt.figure(figsize=(15, 9))

ax = fig.add_subplot(111, xlim=(-2, 3), ylim=(0, 1), aspect='equal')
ax.grid(ls=':')
plt.xlabel(r'$x/L$')
plt.ylabel(r'$\phi/\phi_0$')

line1, = ax.plot([], [], 'o-', lw=1, color='C1')

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)


########################################################################################################

#####     ================      Animation      ================      #####

def init():
    line1.set_data([], [])
    time_text.set_text('')
    return line1, time_text


def animate(frame):
    line1.set_data(x_array, F[frame])
    time_text.set_text(time_template % (t_array[frame]))

    return line1, time_text


anim = ani.FuncAnimation(fig, animate, nt, interval=1, blit=True, init_func=init, repeat_delay=3000)

# ani.save('Pendulum.mp4', fps=15)

plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
plt.show()

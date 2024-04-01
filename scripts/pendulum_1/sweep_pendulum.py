import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import physicsim.pendulum_1 as pendulum_1

from matplotlib.animation import FuncAnimation
from numpy import sin, cos
from scipy.integrate import odeint
from timeit import default_timer as timer
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def sweep_param(var_x, var_y, parameter):
    print(var_x.shape, var_y.shape, parameter.shape)

    Tend = var_x[-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.axis([-Tend * 0.1, Tend * 1.1, -np.pi * 1.1, np.pi * 1.1])
    ax.grid(ls=":")

    line1, = ax.plot([], [], '-', lw=1, color='C1')

    mu_template = r'$\vartheta_0 = %.3f$'
    mu_text = ax.text(0.02, 0.96, '', fontsize=15, transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        mu_text.set_text(mu_template % (parameter[0]))
        return line1, mu_text

    def animate(idx):
        line1.set_data([var_x], [var_y[idx, :]])
        mu_text.set_text(mu_template % (np.degrees(parameter[idx])))
        return line1, mu_text

    _ = FuncAnimation(fig, animate, n_sim, interval=20, blit=True, init_func=init, repeat_delay=5000)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    setup = {"t_sim": 40., "fps": 30, "slowdown": 1., "oversample": 1}
    params = {"g": 9.81, "l": 1., "D": 0., "m": 12.}
    initials = {"phi": np.radians(40.), "om": 0.}

    sim = pendulum_1.Pendulum(setup, params, initials)

    n_sim = 100
    n_steps = sim.n_steps

    phi_initials = np.radians(np.linspace(170., 179, n_sim))
    A = np.zeros((n_sim, 2, 1+n_steps))

    for i in tqdm(range(n_sim)):    
        initials['phi'] = np.radians(170 + i * 0.1)
        sim = pendulum_1.Pendulum(setup, params, initials)
        sim.solve_ode(verbose=False)
        A[i] = sim.full_series

    phi = A[:, 0, :]
    om = A[:, 1, :]

    x, vx = sim.l * sin(phi), sim.l * om * cos(phi)
    y, vy = -sim.l * (cos(phi)), sim.l * om * sin(phi)
    v = np.hypot(vx, vy)

    sweep_param(sim.full_t, phi, phi_initials)

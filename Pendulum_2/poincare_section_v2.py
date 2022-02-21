# Vincent Degrooff - 2022 - EPL
# Interactive Poincare section of the double pendulum

import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, current_process  # , cpu_count
from numpy import sin, cos, pi, arccos, sqrt, degrees, radians
from time import perf_counter
from tqdm import tqdm

from Pendule_double_adim import db_pendulum_solver, see_animation, draw_image
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

ftSz1, ftSz2, ftSz3 = 18, 15, 13
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def dynamics(_, y_):
    phi1, w1, phi2, w2 = y_
    Cos, Sin = cos(phi1 - phi2), sin(phi1 - phi2)

    f1 = (MU * Cos * (sin(phi2) - w1 * w1 * Sin) - L * MU * w2 * w2 * Sin - sin(phi1)) / (1. - MU * Cos * Cos)
    f2 = Sin * (cos(phi1) + w1 * w1 + L * MU * w2 * w2 * Cos) / (L - L * MU * Cos * Cos)

    return np.array([w1, f1, w2, f2])


def jac(_, y_):
    phi1, w1, phi2, w2 = y_
    sin1, sin2 = sin(phi1), sin(phi2)
    cos1, cos2 = cos(phi1), cos(phi2)
    om1, om2 = w1 * w1, w2 * w2
    Cos, Sin = cos(phi1 - phi2), sin(phi1 - phi2)

    num2 = cos1 + om1 + L * MU * om2 * Cos
    num = 2 * MU * Cos * Sin
    den = 1. - MU * Cos * Cos

    f1 = (MU * Cos * (sin2 - om1 * Sin) - L * MU * om2 * Sin - sin1) / den
    f2 = Sin * num2 / (L * den)

    J = np.zeros((4, 4))

    J[0, 1] = 1.

    J[1, 0] = - (MU * Sin * (sin2 - om1 * Sin) + MU * Cos * Cos * om1 + MU * L * om2 * Cos + cos1 + num * f1) / den
    J[1, 1] = - num * w1 / den
    J[1, 2] = MU * (Sin * (sin2 - om1 * Sin) + Cos * (cos2 + om1 * Cos) + L * om2 * Cos) / den + num * f1 / den
    J[1, 3] = - 2 * L * MU * Sin * w2 / den

    J[2, 3] = 1.

    J[3, 0] = (Cos * num2 - Sin * (sin1 + L * MU * om2 * Sin)) / (L * den) - num * f2 / den
    J[3, 1] = 2 * w1 * Sin / (L * den)
    J[3, 2] = (-Cos * num2 + L * MU * om2 * Sin * Sin) / (L * den) + num * f2 / den
    J[3, 3] = num * w2 / den
    return J


def event_phi1(_, y_):
    return sin(y_[0] / 2.)


def event_phi2(_, y_):
    return sin(y_[2] / 2.)


def compute_intersections(U0, max_iter=50, display_info=True):
    global last_run
    if display_info:
        print(current_process())

    my_event = event_phi2 if mode == 1 else event_phi1
    iter_nb, steps = 0, 0
    events = np.zeros((0, 4))
    u_zero = U0

    st = perf_counter()
    while (len(events) < n_points) and (iter_nb < max_iter):
        sol = solve_ivp(dynamics, [0, 200.], u_zero, method='LSODA', events=my_event, jac=jac, rtol=rtol, atol=atol)
        # sol = solve_ivp(dynamics, [0, 200.], u_zero, method='RK45', events=my_event, atol=1e-8, rtol=1.e-8)
        events = np.r_[events, sol.y_events[0]]
        u_zero = sol.y[:, -1]
        iter_nb += 1
        steps += len(sol.t)
        last_run = sol.y[:, -1]

    fn = perf_counter()

    # t, phi1, om1, phi2, om2 = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    if display_info:
        print("LSODA: {:f} s - {:3d} intersections - {:5d} steps needed\n".format(fn - st, events.shape[0], steps))

    # idx_om_a, idx_om_b = (3, 1) if mode == 1 else (1, 3)
    # mask = events[:, idx_om_a] >= 0.
    if mode == 1:
        mask = L * events[:, 3] + events[:, 1] * cos(events[:, 0] - events[:, 2]) > 0.
    else:
        mask = events[:, 1] + MU * L * events[:, 3] * cos(events[:, 0] - events[:, 2]) > 0.

    points = events[mask][:, [0, 1, 3]] if mode == 1 else events[mask][:, [2, 3, 1]]
    # points = events[:, [0, 1, 3]] if mode == 1 else events[:, [2, 3, 1]]

    wrapped_angles = np.fmod(np.fmod(points[:, 0], 2 * pi) + 2 * pi + pi, 2 * pi) - pi
    return np.c_[wrapped_angles, points[:, 1:]]


def get_U0(phi, om, which='positive'):
    # E in terms of g * (m1+m2) * l2 / 2
    # phi1, om1 or phi2, om2

    factor = 1. if which == 'positive' else -1.
    Cos = cos(phi)

    if mode == 1:
        Delta = om * om * (MU * Cos * Cos - 1) + 2 * (Cos - 1) + 2 * E
        if Delta < 0:
            return None
        om2 = -om * Cos / L + factor * sqrt(Delta / MU) / L
        return np.array([phi, om, 0., om2])

    elif mode == 2:
        Delta = om * om * L * L * MU * (MU * Cos * Cos - 1) + 2 * L * MU * (Cos - 1) + 2 * E
        if Delta < 0:
            return None
        om1 = -L * MU * om * Cos + factor * sqrt(Delta)
        return np.array([0., om1, phi, om])


def handle_new_point(event, list_U0, lines):
    phi, om = event.xdata, event.ydata
    U0 = get_U0(phi, om)
    if U0 is None:
        return
    print("\n{:.5f}, {:.5f}, {:.5f}, {:.5f}".format(U0[0], U0[1], U0[2], U0[3]))
    list_U0.append(U0.copy())

    points = compute_intersections(U0, display_info=True)
    if do_save:
        save_file(points, merge=True)

    if use_colors:
        scat = ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=2, c=points[:, 2], cmap=cmap)
        lines.append(scat)
    else:
        line, = ax.plot(points[:, 0], points[:, 1], '.', markersize=1, color='black')
        lines.append(line)


def on_click(event, list_U0, lines):
    print(event.button)
    if event.inaxes != ax:
        return
    if event.button == 1:
        handle_new_point(event, list_U0, lines)

    elif event.button == 2:
        while len(lines) > 0:
            scat = lines.pop()
            scat.remove()
        list_U0.clear()

    # elif event.button == 3:
    #     print("ohé")
    #     phi, om = event.xdata, event.ydata
    #     U0 = get_U0(phi, om)
    #     if U0 is None:
    #         return
    #     see_animation(*db_pendulum_solver(U0, L, MU, Tend=20, l1=1, m1=1.))

    elif event.button == 3:
        if len(lines) > 0:
            scat = lines.pop()
            scat.remove()
            list_U0.pop()
    fig.canvas.draw()


def on_touch(event, list_U0, lines):
    if (event.key == 'a') and (0 < len(list_U0)):  # start a simulation with the parameters from the last run
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), L, MU, Tend=20, l1=1, m1=1.))

    elif (event.key == 'z') and (0 < len(list_U0)):  # same with a dialog box for parameters
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), L, MU, mode='animation_ask'))

    elif (event.key == 'd') and (0 < len(list_U0)):  # display an image of the path
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), L, MU, Tend=20, sps=720, l1=1., m1=1.))

    elif (event.key == 'e') and (0 < len(list_U0)):  # same with a dialog box for parameters
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), L, MU, sps=720, mode='draw_ask'))

    elif (event.key == 'c') and (0 < len(list_U0)):  # continue the process with the last trajectory
        points = compute_intersections(last_run, display_info=False)
        line, = ax.plot(points[:, 0], points[:, 1], '.', markersize=1., color='black')
        lines.append(line)
        if do_save:
            save_file(points, merge=True)
        fig.canvas.draw()

    elif event.key == 'r':  # compute intersections with multiple random initial conditions
        # plot_random_multiprocess(6, 10, list_U0)
        plot_random(n_samples=10, U0_list=list_U0)
        fig.canvas.draw()

    elif event.key == 'g':  # display points "uniformly" spread around the section
        for i in range(200):
            var_1, var_2 = pick_random(random_list)
            ax.plot([var_1], [var_2], 'o', markersize=3, color='C0')
        fig.canvas.draw()

    elif event.key == 'x':  # start simulation at coordinates of the mouse
        phi, om = event.xdata, event.ydata
        U0 = get_U0(phi, om)
        if U0 is None:
            return
        see_animation(*db_pendulum_solver(U0, L, MU, Tend=20, l1=1, m1=1.))

    elif event.key == 'n':  # new series of points
        handle_new_point(event, list_U0, lines)
        fig.canvas.draw()


def create_fig():
    fig_, ax_ = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    ax_.set_xlabel(r'$\varphi_{:d} \quad [\;rad\;]$'.format(mode))
    ax_.set_ylabel(r'$\dot \varphi_{:d} \quad [\;rad/s\;]$'.format(mode))
    ax_.tick_params(axis='both', which='major', labelsize=8)

    deg2rad = lambda angle: radians(angle)
    rad2deg = lambda angle: degrees(angle)
    deg2rad_ = lambda angle: radians(angle / w)
    rad2deg_ = lambda angle: degrees(angle * w)

    secax_x = ax_.secondary_xaxis('top', functions=(rad2deg, deg2rad))
    secax_y = ax_.secondary_yaxis('right', functions=(rad2deg_, deg2rad_))
    for secax in [secax_x, secax_y]:
        secax.tick_params(axis='both', which='major', labelsize=8)
        # secax.set_xlabel(r'$[\;^{\circ}\;]$')

    ax_.text(0.025, 0.95, r'$E       = {:.2f} $'.format(E), fontsize=10, wrap=True, transform=ax_.transAxes)
    ax_.text(0.025, 0.9, r'$\lambda = {:.2f} $'.format(L), fontsize=10, wrap=True, transform=ax_.transAxes)
    ax_.text(0.025, 0.85, r'$\mu     = {:.2f} $'.format(MU), fontsize=10, wrap=True, transform=ax_.transAxes)

    x = np.linspace(-phi_max * 0.99, phi_max * 0.99, 500)
    if mode == 1:
        y = sqrt(2 * (cos(x) - 1 + E) / (1 - MU * cos(x) * cos(x)))
    else:
        y = sqrt(2 * (L * MU * (cos(x) - 1) + E) / (L * L * MU * (1 - MU * cos(x) * cos(x))))

    ax_.plot(x, y, color='black', alpha=0.5)
    ax_.plot(x, -y, color='black', alpha=0.5)
    bounds_interpolated = interp1d(x, y, kind='cubic')

    return fig_, ax_, bounds_interpolated


def interactive_func():

    initConditions = []
    lines = []

    # fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, initConditions, lines))
    fig.canvas.mpl_connect('key_press_event', lambda event: on_touch(event, initConditions, lines))
    print("\nHelp menu :\n\
    -> n : start a new computation using the mouse position and draws the points on the Poincare section\n\
    -> c : continue to compute intersection points with the last trajectory\n\
    -> x : start an animation of the double pendulum using the mouse position as initial condition\n\
    -> a : start an animation of the double pendulum using the last run of the Poincare section\n")
    return


# def plot_random_one_thread(n_samples, U0_list):
#     init_conditions = []
#
#     for i in range(n_samples):
#         phi = 0.99 * phi_max * np.random.uniform(-1, 1)
#         om = f_y(phi) * np.random.uniform(-1, 1)
#         u_zero = get_U0(phi, om)
#         init_conditions.append(u_zero)
#         U0_list.append(u_zero)
#
#     points_list = []
#     st = perf_counter()
#     for U0 in init_conditions:
#         points_list.append(compute_intersections(U0))
#     print("Time taken UNITHREAD : {:.2f} s".format(perf_counter() - st))
#
#     for points in points_list:
#         ax.plot(points[:, 0], points[:, 1], 'o', markersize=5., alpha=0.4, color='black')
#         # scat = ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=1, c=points[:, 2], cmap=cmap)
#
#     return

def create_random(nx, ny):
    my_list = []
    dx, dy = phi_max / nx, om_max / ny
    current_x = dx  # -0.99 * phi_max + dx
    if mode == 1:
        bound_om = lambda x: sqrt(2 * (cos(x) - 1 + E) / (1 - MU * cos(x) * cos(x)))
    else:
        bound_om = lambda x: sqrt(2 * (L * MU * (cos(x) - 1) + E) / (L * L * MU * (1 - MU * cos(x) * cos(x))))
    while current_x < 0.99 * phi_max:
        current_y = -bound_om(current_x)
        while current_y < om_max:
            my_list.append((current_x - dx, current_x, current_y, current_y + dy))
            current_y += dy
        current_x += dx

    return my_list


def pick_random(rd_list, sample=-1):
    if mode == 1:
        check = lambda phi, om: om * om * (MU * cos(phi) * cos(phi) - 1) + 2 * (cos(phi) - 1) + 2 * E
    else:
        check = lambda phi, om: om * om * L * L * MU * (MU * cos(phi) ** 2 - 1) + 2 * L * MU * (cos(phi) - 1) + 2 * E

    valid, i = False, 0
    rd_phi, rd_om = 0., 0.
    while not valid and i < 10:
        if sample == -1:
            rd_box = rd_list[np.random.randint(0, len(rd_list))]
        else:
            rd_box = rd_list[sample % len(rd_list)]
        rd_phi = np.random.uniform(rd_box[0], rd_box[1])
        rd_om = np.random.uniform(rd_box[2], rd_box[3])
        valid = check(rd_phi, rd_om) > 0.
        i += 1
    return rd_phi, rd_om


def plot_random_multiprocess(n_threads, n_samples):
    p = Pool(n_threads)
    init_conditions = []

    for i in range(n_samples):
        # phi = 0.99 * phi_max * np.random.uniform(-1, 1)
        # om = f_y(phi) * np.random.uniform(-1, 1)
        phi, om = pick_random(random_list, i)
        u_zero = get_U0(phi, om, 'positive')
        init_conditions.append(u_zero)

    work = tuple([*init_conditions])
    st = perf_counter()
    points_list = p.map(compute_intersections, work)
    # with Pool(n_threads) as p:
    #     list(tqdm(p.imap(compute_intersections, work), total=n_samples))
    #     points_list = tqdm(p.imap(compute_intersections, work), total=n_samples)

    print("\nTime taken : {:.2f} s".format(perf_counter() - st))

    for points in points_list:
        ax.plot(points[:, 0], points[:, 1], '.', markersize=1., color='black')
        # ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=1, c=points[:, 2], cmap=cmap)
    if do_save:
        save_file(np.vstack(points_list), merge=True)

    return


def plot_random(n_samples, U0_list):
    points_list = []
    
    st = perf_counter()
    for _ in tqdm(range(n_samples)):
        # phi = 0.99 * phi_max * np.random.uniform(-1, 1)
        # om = f_y(phi) * np.random.uniform(-1, 1)
        phi, om = pick_random(random_list)
        u_zero = get_U0(phi, om, 'positive')
        U0_list.append(u_zero)
        points_list.append(compute_intersections(u_zero))

    print("\nTime taken : {:.2f} s".format(perf_counter() - st))

    points = np.vstack(points_list)
    ax.plot(points[:, 0], points[:, 1], '.', markersize=1., color='black')
    # ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=1, c=points[:, 2], cmap=cmap)
    if do_save:
        save_file(points, merge=True)

    return points


def get_bounds():
    if mode == 1:  # phi1 vs om1 when phi2=0
        w_max = sqrt(2 * E / (1 - MU))  # om = om1
        w_o_min = -sqrt(2 * E / (1 - MU)) / L  # om_o = om2
        w_o_max = sqrt(2 * E / ((1 - MU) * MU)) / L
        angle_max = arccos(1 - E) if E < 2 else pi  # phi = phi1
    else:  # phi2 vs om2 when phi1=0
        w_o_min = -sqrt(2 * E * MU / (1 - MU))
        w_o_max = sqrt(2 * E / (1 - MU))
        w_max = 1 / L * sqrt(2 * E / (MU * (1 - MU)))
        angle_max = arccos(1 - E / (MU * L)) if E < 2 * MU * L else pi  # phi = phi1
    return angle_max, w_max, w_o_min, w_o_max


def save_file(points, merge=False):
    if merge and reuse > 0:
        filename = f"./Data/coordinates_{series}{reuse}.txt"
        with open(filename, "a") as txt_file:
            np.savetxt(txt_file, points, fmt='%.5e')
    else:
        i = 1
        while os.path.exists(f"./Data/coordinates_{i}.txt"):
            i += 1
        filename = f"./Data/coordinates_{i}.txt"
        head = "{:e} {:e} {:e} {:d}".format(E, L, MU, mode)
        np.savetxt(filename, points, header=head, fmt='%.5e')
    return


def load_file():
    global E, L, MU, mode
    filename = f"./Data/coordinates_{series:s}{reuse:d}.txt"
    # print(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError

    with open(filename, "r") as txt_file:
        E, L, MU, mode = [float(x) for x in txt_file.readline().strip().split(" ")[1:]]
        mode = int(mode)

    return np.loadtxt(filename, skiprows=1)


# general parameters: must be outside "if __name__..." for multiprocessing
mode = 2  # Poincare section in the (phi_mode, om_mode) plane
L = 1.0   # L = l2 / l1
MU = 0.5  # MU = m2 / (m1 + m2)
E = 2.    # energy level

N = 4            # number of orbits to compute
n_points = 1000  # number of points on the section for each orbit
rtol, atol = 1.e-9, 1.e-8


if __name__ == "__main__":

    # parameters used for simulation of a particular orbit
    g, l1 = 9.81, 1.0
    w = sqrt(g / l1)
    cmap = plt.get_cmap('jet_r')

    # 1, 4, 6, 8, b4
    reuse, series = 4, 'b'  # append computations to ./Data/coordinates_{reuse}.txt
    interactive = True  # compute trajectories by clicking on the section
    do_save = False  # save the results to a file
    use_colors = True  # represent the other angular velocity with colors

    if mode != 1 and mode != 2:
        raise EnvironmentError

    # needed now since it changes the global variables
    res = load_file() if reuse > 0 else []

    phi_max, om_max, om_o_min, om_o_max = get_bounds()
    normalize = plt.Normalize(vmin=om_o_min, vmax=om_o_max)
    fig, ax, f_y = create_fig()
    random_list = create_random(5, 5)
    # print("Number of cpu : ", cpu_count())

    if interactive:
        if reuse > 0:
            ax.plot(res[:, 0], res[:, 1], '.', markersize=0.5, color='black')
        elif do_save:
            save_file(np.zeros((0, 3)), merge=False)
        last_run = np.zeros(4)
        interactive_func()

    else:
        if reuse > 0:
            if not use_colors:
                ax.plot(res[:, 0], res[:, 1], '.', markersize=0.5, color='black')
            else:
                ax.scatter(res[:, 0], res[:, 1], norm=normalize, s=1, c=res[:, 2], cmap=cmap)

            # fig.savefig(f"./Sections/run_{series}{reuse}.svg", format='svg', bbox_inches='tight')

        # if input("Sample new points with multiprocess ? [y/n]") == 'y':
            # # res = plot_random(n_samples=N, U0_list=[])
            # plot_random_multiprocess(n_threads=4, n_samples=N)

    plt.show()

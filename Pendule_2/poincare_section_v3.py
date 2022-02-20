# Vincent Degrooff - 2022 - EPL
# Interactive Poincare section of the double pendulum - computes both phi1=0 and phi2=0 intersections

import os
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count, current_process
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


event_phi1 = lambda _, y_: sin(y_[0] / 2.)
event_phi2 = lambda _, y_: sin(y_[2] / 2.)


def compute_intersections(U0, max_iter=50, display_info=True):
    global last_run
    if display_info:
        print(current_process())

    iter_nb, steps = 0, 0
    events_A, events_B = np.zeros((0, 4)), np.zeros((0, 4))
    last_run = U0

    st = perf_counter()
    while ((len(events_A) < n_points) and (len(events_B) < n_points)) and (iter_nb < max_iter):
        # noinspection PyTypeChecker
        # sol = solve_ivp(dynamics, [0, 1000], U0, args=(l, m), method='RK45', events=my_event, rtol=1.e-8)
        sol = solve_ivp(dynamics, [0, 200.], last_run, method='LSODA', jac=jac, rtol=1.e-10, atol=1e-6,
                        events=(event_phi1, event_phi2))

        events_A = np.r_[events_A, sol.y_events[1]]
        events_B = np.r_[events_B, sol.y_events[0]]

        iter_nb += 1
        steps += len(sol.t)
        last_run = sol.y[:, -1]

    fn = perf_counter()
    # t, phi1, om1, phi2, om2 = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    if display_info:
        s = "LSODA: {:f} s - ({:3d}, {:3d}) intersections (A,B) - {:5d} steps needed\n"
        print(s.format(fn - st, events_A.shape[0], events_B.shape[0], steps))

    # idx_om_a, idx_om_b = (3, 1) if mode == 1 else (1, 3)
    # mask = events[:, idx_om_a] >= 0.
    mask_A = L * events_A[:, 3] + events_A[:, 1] * cos(events_A[:, 0] - events_A[:, 2]) > 0.
    mask_B = events_B[:, 1] + MU * L * events_B[:, 3] * cos(events_B[:, 0] - events_B[:, 2]) > 0.

    points_A = events_A[mask_A][:, [0, 1, 3]]
    points_B = events_B[mask_B][:, [2, 3, 1]]
    # points = events[:, [0, 1, 3]] if mode == 1 else events[:, [2, 3, 1]]

    new_angles_A = np.fmod(np.fmod(points_A[:, 0], 2 * pi) + 2 * pi + pi, 2 * pi) - pi
    new_angles_B = np.fmod(np.fmod(points_B[:, 0], 2 * pi) + 2 * pi + pi, 2 * pi) - pi

    return np.c_[new_angles_A, points_A[:, 1:]], np.c_[new_angles_B, points_B[:, 1:]]


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


def on_click(event, list_U0, lines):
    if event.inaxes != ax:
        return
    if event.button == 1:
        phi, om = event.xdata, event.ydata
        U0 = get_U0(phi, om)
        if U0 is None:
            return
        print("\n{:.5f}, {:.5f}, {:.5f}, {:.5f}".format(U0[0], U0[1], U0[2], U0[3]))
        list_U0.append(U0.copy())

        points_A, points_B = compute_intersections(U0, display_info=True)
        points = points_A if mode == 1 else points_B
        if do_save:
            save_file_new(points_A, points_B, file_idx)

        if use_colors:
            line = ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=2, c=points[:, 2], cmap=cmap)
        else:
            line, = ax.plot(points[:, 0], points[:, 1], 'o', markersize=.5, color='black')
        lines.append(line)

    elif event.button == 2:
        while len(lines) > 0:
            scat = lines.pop()
            scat.remove()
        list_U0.clear()

    elif event.button == 3:
        if len(lines) > 0:
            scat = lines.pop()
            scat.remove()
            list_U0.pop()
    fig.canvas.draw()


def on_touch(event, list_U0, lines):
    if (event.key == 'a') and (0 < len(list_U0)):
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), L, MU, Tend=20, l1=1, m1=1.))

    elif (event.key == 'd') and (0 < len(list_U0)):
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), L, MU, Tend=20, sps=720, l1=1., m1=1.))

    elif (event.key == 'z') and (0 < len(list_U0)):
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), L, MU, mode='animation_ask'))

    elif (event.key == 'e') and (0 < len(list_U0)):
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), L, MU, sps=720, mode='draw_ask'))

    elif (event.key == 'c') and (0 < len(list_U0)):
        points_A, points_B = compute_intersections(last_run, display_info=False)
        points = points_A if mode == 1 else points_B
        if do_save:
            save_file_new(points_A, points_B, file_idx)

        line, = ax.plot(points[:, 0], points[:, 1], 'o', markersize=.5, color='black')
        lines.append(line)
        fig.canvas.draw()

    elif event.key == 'r':
        # plot_random_multiprocess(6, 10, list_U0)
        plot_random(n_samples=10, U0_list=list_U0)
        fig.canvas.draw()

    elif event.key == 'g':
        for i in range(200):
            var_1, var_2 = pick_random(random_list)
            ax.plot([var_1], [var_2], 'o', markersize=3, color='C0')
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
        secax.set_xlabel(r'$[\;^{\circ}\;]$')

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

    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, initConditions, lines))
    fig.canvas.mpl_connect('key_press_event', lambda event: on_touch(event, initConditions, lines))
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
    current_x = -0.99 * phi_max + dx
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


def pick_random(rd_list):
    if mode == 1:
        check = lambda phi, om: om * om * (MU * cos(phi) * cos(phi) - 1) + 2 * (cos(phi) - 1) + 2 * E
    else:
        check = lambda phi, om: om * om * L * L * MU * (MU * cos(phi) ** 2 - 1) + 2 * L * MU * (cos(phi) - 1) + 2 * E

    valid, i = False, 0
    rd_phi, rd_om = 0., 0.
    while not valid and i < 10:
        rd_box = rd_list[np.random.randint(0, len(rd_list))]
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
        phi, om = pick_random(random_list)
        u_zero = get_U0(phi, om, 'positive')
        init_conditions.append(u_zero)

    work = tuple([*init_conditions])
    st = perf_counter()
    points_list = p.map(compute_intersections, work)
    # with Pool(n_threads) as p:
    #     list(tqdm(p.imap(compute_intersections, work), total=n_samples))
    #     points_list = tqdm(p.imap(compute_intersections, work), total=n_samples)

    print("\nTime taken : {:.2f} s".format(perf_counter() - st))

    points_A = np.vstack([tuple_points[0] for tuple_points in points_list])
    points_B = np.vstack([tuple_points[1] for tuple_points in points_list])
    if do_save:
        save_file_new(points_A, points_B, file_idx)

    points = points_A if mode == 1 else points_B
    ax.plot(points[:, 0], points[:, 1], 'o', markersize=.5, color='black')
    # ax.scatter(points[:, 0], points[:, 1], norm=normalize, s=1, c=points[:, 2], cmap=cmap)
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
        points_list.append(compute_intersections(u_zero, display_info=False))

    print("\nTime taken : {:.2f} s".format(perf_counter() - st))

    points_A = np.vstack([tuple_points[0] for tuple_points in points_list])
    points_B = np.vstack([tuple_points[1] for tuple_points in points_list])
    if do_save:
        save_file_new(points_A, points_B, file_idx)

    points = points_A if mode == 1 else points_B
    ax.plot(points[:, 0], points[:, 1], 'o', markersize=.5, color='black')

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


def save_file_new(pointsA, pointsB, number):
    if os.path.exists(f"./Coordinates_A/run_{number:d}_A.txt"):
        with open(f"./Coordinates_A/run_{number:d}_A.txt", "a") as txt_file:
            np.savetxt(txt_file, pointsA, fmt='%.5e')
        with open(f"./Coordinates_B/run_{number:d}_B.txt", "a") as txt_file:
            np.savetxt(txt_file, pointsB, fmt='%.5e')
    else:
        head = "{:e} {:e} {:e}".format(E, L, MU)
        np.savetxt(f"./Coordinates_A/run_{number:d}_A.txt", pointsA, header=head, fmt='%.5e')
        np.savetxt(f"./Coordinates_B/run_{number:d}_B.txt", pointsB, header=head, fmt='%.5e')

    return


def load_file_new(number):
    global E, L, MU
    filenameA = f"./Coordinates_A/run_{number:d}_A.txt"
    filenameB = f"./Coordinates_B/run_{number:d}_B.txt"
    with open(filenameA, "r") as txt_file:
        E, L, MU = [float(x) for x in txt_file.readline().strip().split(" ")[1:]]

    if mode == 1:
        return np.loadtxt(filenameA, skiprows=1)
    else:
        return np.loadtxt(filenameB, skiprows=1)


def get_file_index(number):
    if number > 0:
        return number, load_file_new(number)
    i = 1
    while os.path.exists(f"./Coordinates_A/run_{i:d}_A.txt"):
        i += 1
    if do_save:
        save_file_new(np.zeros((0, 3)), np.zeros((0, 3)), i)
    return i, np.zeros((0, 3))


# must be outside "if __name__..." for multiprocessing
L, MU = 2., 2./3.
E = 2.1
mode = 2
n_points = 1000
N = 10

if __name__ == "__main__":
    # m1, m2, l1, l2 = 0.2, 5., 0.5, 0.5
    g, l1 = 9.81, 1.0
    w = sqrt(g / l1)
    cmap = plt.get_cmap('jet_r')

    interactive = False
    do_save = False
    file_idx, res = get_file_index(1)

    use_colors = False
    if mode != 1 and mode != 2:
        raise EnvironmentError

    # needed now since it changes the global variables
    # res = load_file_new(file_idx)

    phi_max, om_max, om_o_min, om_o_max = get_bounds()
    normalize = plt.Normalize(vmin=om_o_min, vmax=om_o_max)
    fig, ax, f_y = create_fig()
    random_list = create_random(50, 50)
    print("Number of cpu : ", cpu_count())

    if interactive:
        if len(res) > 0:
            ax.plot(res[:, 0], res[:, 1], '.', markersize=0.5, color='black')
        last_run = np.zeros(4)
        interactive_func()
        plt.show(block=True)

    else:
        if len(res) > 0:
            if not use_colors:
                ax.plot(res[:, 0], res[:, 1], '.', markersize=0.5, color='black')
            else:
                ax.scatter(res[:, 0], res[:, 1], norm=normalize, s=1, c=res[:, 2], cmap=cmap)

        if input("Sample new points with multiprocess ? [y/n]") == 'y':
            # res = plot_random(n_samples=N, U0_list=[])
            plot_random_multiprocess(n_threads=4, n_samples=N)
        plt.show()

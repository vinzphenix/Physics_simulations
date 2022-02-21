import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, arccos, array, zeros, sqrt, linspace, degrees, radians, remainder

from Pendule_double_adim import db_pendulum_solver, see_animation, draw_image
from scipy.interpolate import interp1d

ftSz1, ftSz2, ftSz3 = 18, 15, 13
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def interpolate(Ua, Ub):
    phi1_a, om1_a, phi2_a, om2_a = Ua
    phi1_b, om1_b, phi2_b, om2_b = Ub
    if mode == 1:
        coef = phi2_a / (phi2_a - phi2_b)
    else:
        coef = phi1_a / (phi1_a - phi1_b)

    U_inter = Ua * (1 - coef) + Ub * coef
    return U_inter


def runge(dt, U0, nSteps):

    def f(u):
        phi1, om1, phi2, om2 = u
        Cos, Sin = cos(phi1 - phi2), sin(phi1 - phi2)

        f1 = (m * Cos * (sin(phi2) - om1 * om1 * Sin) - l * m * om2 * om2 * Sin - sin(phi1)) / (1. - m * Cos * Cos)
        f2 = Sin * (cos(phi1) + om1 * om1 + l * m * om2 * om2 * Cos) / (l - l * m * Cos * Cos)

        return array([om1, f1, om2, f2])

    point_list = zeros((nPoints, 3))
    index = 0

    for i in range(nSteps-1):
        K1 = f(U0)
        K2 = f(U0 + K1 * dt / 2)
        K3 = f(U0 + K2 * dt / 2)
        K4 = f(U0 + K3 * dt)
        U1 = U0 + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

        phi1_pr, om1_pr, phi2_pr, om2_pr = U0
        phi1_nx, om1_nx, phi2_nx, om2_nx = U1

        if mode == 1:
            condition_angle = (sin(phi2_pr) * sin(phi2_nx) <= 0.) and (cos(phi2_pr) > 0.)
            condition_velocity = 0. <= l * om2_nx + om1_nx * cos(phi1_nx)
        else:
            condition_angle = (sin(phi1_pr) * sin(phi1_nx) <= 0.) and (cos(phi1_pr) > 0.)
            condition_velocity = (0. <= om1_nx + m * l * om2_nx * cos(phi2_nx))

        if condition_angle and condition_velocity:
            phi1_pr, phi2_pr = remainder(U0[[0, 2]] + pi, 2 * pi) - pi
            phi1_nx, phi2_nx = remainder(U1[[0, 2]] + pi, 2 * pi) - pi

            _1, _2, _3, _4 = interpolate(array([phi1_pr, om1_pr, phi2_pr, om2_pr]),
                                     array([phi1_nx, om1_nx, phi2_nx, om2_nx]))
            #z = 2 * m * l * (1 - cos(_3))
            #z = _2*_2 + m*l*l * _4*_4 * 2*m*l * _2*_4 * cos(_3)
            if mode == 1:
                point_list[index] = _1, _2, _4
            else:
                point_list[index] = _3, _4, _2
            index += 1

            if index == nPoints:
                return index, point_list
        U0 = U1
    print("Not enough", index, nPoints, nSteps)
    return index, point_list


def get_U0(phi, om):
    # E in terms of g * (m1+m2) * l2 / 2
    # phi1, om1 or phi2, om2
    Cos = cos(phi)

    if mode == 1:
        Delta = om*om * (m*Cos*Cos - 1) + 2 * (Cos - 1) + E * (1 - m)
        if Delta < 0:
            return None
        om2 = -om * Cos / l + sqrt(Delta/m) / l
        return array([phi, om, 0., om2])

    elif mode == 2:
        Delta = om*om * l * l * m * (m * Cos * Cos - 1) + 2 * l * m * (Cos - 1) + E
        if Delta < 0:
            return None
        om1 = -l*m * om * Cos + sqrt(Delta)
        return np.array([0., om1, phi, om])


def sampleRandom(nb):
    for i in range(nb):
        phi = 0.99 * phi_max * (2 * np.random.rand() - 1)
        om = f_y(phi) * (2 * np.random.rand() - 1)
        U0 = get_U0(phi, om)
        nbPoints, points = runge(delta_t, U0.copy(), nMax)
        ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], 'o', markersize=0.5, color='black')
        # scat = ax.scatter(points[:nbPoints, 0], points[:nbPoints, 1], norm=normalize,
        #                  s=1, c=points[:nbPoints, 2], cmap=cmap)


def on_click(event, list_U0):
    if event.inaxes != ax:
        return
    if event.button == 1:
        phi, om = event.xdata, event.ydata
        U0 = get_U0(phi, om)
        if U0 is None:
            return
        print("{:.5f}, {:.5f}, {:.5f}, {:.5f}".format(U0[0], U0[1], U0[2], U0[3]))
        list_U0.append(U0.copy())

        nbPoints, points = runge(delta_t, U0.copy(), nMax)

        #line, = ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], color=cmap())
        if use_colors:
            scat = ax.scatter(points[:nbPoints, 0], points[:nbPoints, 1], norm=normalize,
                          s=2, c=points[:nbPoints, 2], cmap=cmap)
            lines.append(scat)
        else:
            line, = ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], 'o', markersize=0.5, color='black')
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
    fig1.canvas.draw()


def on_touch(event, list_U0):
    if (event.key == 'a') and (0 < len(list_U0)):
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), l, m, Tend=20, l1=1, m1=1.))

    elif (event.key == 'd') and (0 < len(list_U0)):
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), l, m, Tend=20, sps=720, l1=1., m1=1.))

    elif (event.key == 'z') and (0 < len(list_U0)):
        see_animation(*db_pendulum_solver(list_U0[-1].copy(), l, m, mode='animation_ask'))

    elif (event.key == 'e') and (0 < len(list_U0)):
        draw_image(*db_pendulum_solver(list_U0[-1].copy(), l, m, sps=720, mode='draw_ask'))

    elif event.key == 'r':
        for i in range(15):
            sampleRandom(1)
            fig1.canvas.draw()


if __name__ == "__main__":

    g = 9.81
    l1 = 1.0
    w = sqrt(g/l1)

    #m1, m2, l1, l2 = 0.2, 5., 0.5, 0.5
    cmap = plt.get_cmap('jet_r')
    mode = 2
    l, m = 1.0, 0.5
    E = 0.2

    interactive = True
    use_colors = False

    nPoints = 200
    nMax = 50000
    delta_t = 0.05

    if mode != 1 and mode != 2:
        raise EnvironmentError

    if mode == 1:  # phi1 vs om1 when phi2=0
        om_max = sqrt(E)  # om = om1
        om_o_min = sqrt(E)/l  # om_o = om2
        om_o_max = 1 / l * sqrt(E / m)
        phi_max = pi  # phi = phi1
        if E < 4 / (1 - m):
            phi_max = arccos(1 - E * (1 - m) / 2)
    else:  # phi2 vs om2 when phi1=0
        om_o_min = sqrt(E * m / (1 - m))
        om_o_max = sqrt(E / (1 - m))
        om_max = 1 / l * sqrt(E / (m * (1 - m)))
        phi_max = pi
        if E < 4 * l * m:
            phi_max = arccos(1 - E / (2 * l * m))

    #filename = "./Data/" + "coordinates_" + "{:.3f}_".format(E) + "{:.3f}_".format(l) + "{:.3f}".format(m) + ".txt"
    #file = open(filename, "a")
    #file.write("{} {} {}\n".format(E, l, m))

    fig1, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
    ax.set_xlabel(r'$\varphi_2 \quad [\;rad\;]$')
    ax.set_ylabel(r'$\dot{\varphi_2} \quad [\;rad/s\;]$')
    ax.tick_params(axis='both', which='major', labelsize=8)

    deg2rad = lambda angle: radians(angle)
    rad2deg = lambda angle: degrees(angle)
    deg2rad_ = lambda angle: radians(angle/w)
    rad2deg_ = lambda angle: degrees(angle*w)

    secax_x = ax.secondary_xaxis('top', functions=(rad2deg, deg2rad))
    secax_y = ax.secondary_yaxis('right', functions=(rad2deg_, deg2rad_))
    for secax in [secax_x, secax_y]:
        secax.tick_params(axis='both', which='major', labelsize=8)
        secax.set_xlabel(r'$[\;^{\circ}\;]$')

    ax.text(0.025, 0.95, r'$E       = {:.2f} $'.format(E), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.025, 0.9, r'$\lambda = {:.2f} $'.format(l), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.025, 0.85, r'$\mu     = {:.2f} $'.format(m), fontsize=10, wrap=True, transform=ax.transAxes)

    x = linspace(-phi_max*0.99, phi_max*0.99, 500)
    if mode == 1:
        y = sqrt(- (E * (1 - m) + 2 * (cos(x) - 1)) / (m * cos(x) * cos(x) - 1))
    else:
        y = sqrt((2 * l * m * (cos(x) - 1) + E) / (l * l * m * (1 - m * cos(x) ** 2)))

    f_y = interp1d(x, y, kind='cubic')

    ax.plot(x, y, color='black', alpha=0.5)
    ax.plot(x, -y, color='black',  alpha=0.5)

    initConditions = []
    lines = []
    normalize = plt.Normalize(vmin=-om_o_min, vmax=om_o_max)
    #normalize = plt.Normalize(vmin=0, vmax=E/2)

    if interactive:
        cid_1 = fig1.canvas.mpl_connect('button_press_event', lambda event: on_click(event, initConditions))
        cid_2 = fig1.canvas.mpl_connect('key_press_event', lambda event: on_touch(event, initConditions))
    else:
        sampleRandom(2)

    plt.show(block=True)

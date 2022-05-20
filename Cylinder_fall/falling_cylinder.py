import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, tan, hypot, radians, linspace, array, arange, amax, amin
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81                  # acceleration due to gravity, in m/s²
a0 = 10.0                 # angle pente    [rad]
R = 0.80                  # rayon roue en  [m]
m = 9.5                  # masse roue     [kg]
M = 1.00                  # masse pendule  [m]
L = 1.00                  # length pendule [m]
C = -14.65                 # couple         [N*m]
D1 = 0.0
D2 = 0.0

th00 = 170               # angle initial pendule
om00 = 0.00               # vitesse angulaire initiale pendule
x0  = 0.00               # position initiale roue
v0  = 1.00                # vitesse initiale roue

Tend = 27.9                 # [s]    -  fin de la simulation

n = int(200*Tend)
fps = 30
ratio = n//(int(Tend*fps))

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

th0, om0, a = radians([90 - th00 - a0, om00, a0])
t = linspace(0, Tend, n)
U0 = array([th0, om0, x0, v0])


def f(u, _):
    # th, om, x, v = u

    dv = (m + M) * g * sin(a) + C / R - (D1 + D2) * u[3] + M * L * u[1] * u[1] * cos(u[0]) + \
         M * sin(u[0]) * (g * cos(a + u[0]) - D2 * (L * u[3] * sin(u[0]) + L * L * u[1]))
    dom = 1 / L * (g * cos(a + u[0]) - D2 * (L * u[3] * sin(u[0]) + L * L * u[1]) + dv * sin(u[0]))

    return array([u[1], dom, u[3], dv])


tic = timer()
sol = odeint(f, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))

th, x = array(sol[:, 0]), array(sol[:, 2])
om, v = array(sol[:, 1]), array(sol[:, 3])
phi = (x - x0) / R

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

xc, yc = x * cos(a), -x * sin(a)  # position du centre du cercle
x1, y1 = xc + R * cos(phi + a), yc - R * sin(phi + a)  # position d'un point sur cercle
x3, y3 = xc - R * cos(phi + a), yc + R * sin(phi + a)  # position d'un point sur cercle
x2, y2 = xc + L * cos(th + a), yc - L * sin(th + a)  # position du pendule

vx2 = +v * cos(a) - L * sin(th + a) * om
vy2 = -v * sin(a) - L * cos(th + a) * om
v2 = hypot(vx2, vy2)

xmin, xmax = amin(xc) - 4.0 * max(R, L), amax(xc) + 4.0 * max(R, L)
ymin, ymax = amin(yc) - 1.5 * max(R, L), amax(yc) + 1.5 * max(R, L)
L_X, L_Y = xmax - xmin, ymax - ymin
#print('xmin = ', xmin) ; print('xmax = ', xmax)
#print('ymin = ', ymin) ; print('ymax = ', ymax)


########################################################################################################

#####     ================      Animation du Système      ================      #####

def see_animation(save=""):
    global ratio, n
    ratio = 1 if save == "snapshot" else ratio
    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    #####     ================      Création de la figure      ================      #####

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211, xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')
    ax2 = fig.add_subplot(223)
    ax2.grid(ls=':')
    ax.grid(ls=':')
    ax3 = fig.add_subplot(224)
    ax3.grid(ls=':')
    ax2.set_xlabel(r'$\theta \: \rm [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$v \: \rm [m/s]$', fontsize=ftSz2)
    ax3.set_xlabel(r'$\omega \: \rm [rad/s]$', fontsize=ftSz2)  # ; ax3.set_ylabel(r'$v \: \rm [m/s]$')

    line1, = ax.plot([], [], 'o-', lw=2, color='grey')
    line2, = ax.plot([], [], 'o-', lw=2, color='orange')
    phase21, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase31, = ax3.plot([], [], marker='o', ms=8, color='C0')

    xd = arange(xmin, xmax, 0.01)
    ax.plot(xd, yc[0] - R / cos(a) - tan(a) * (xd - xc[0]))

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.45, 0.88, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((xmax - L_X * 0.03, ymax - 0.03 * L_X),
                           L_X * 0.02, theta1=90, theta2=90, color='lightgrey')

    circ = patches.Circle((xc[0], yc[0]), radius=R, edgecolor=None, facecolor='lightgrey', lw=4)

    ax.text(0.02, 0.06, r'$\alpha  = {:.2f} \: \rm [^\circ] $'.format(a0), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.02, 0.14, r'$\tau  = {:.2f} \: \rm [N \cdot m]$'.format(abs(C)), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.15, 0.06, r'$M  = {:.2f} \: \rm [kg]$'.format(M), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.15, 0.14, r'$m  = {:.2f} \: \rm [kg]$'.format(m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.28, 0.06, r'$L  = {:.2f} \: \rm [m]$'.format(L), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.28, 0.14, r'$R  = {:.2f} \: \rm [m]$'.format(R), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.72, 0.92, r'$\theta_0  = {:.2f} $'.format(th00), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.72, 0.84, r'$\omega_0  = {:.2f} $'.format(om00), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$x_0  = {:.2f} $'.format(x0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.84, r'$v_0  = {:.2f} $'.format(v0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(th, v, color='C1')
    #ax2.plot(om, v, color='C1')
    ax3.plot(om, v, color='C2')

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        phase21.set_data([], [])
        phase31.set_data([], [])
        time_text.set_text('')
        circ.center = (xc[0], yc[0])
        sector.set_theta1(90)
        return line1, line2, time_text, circ, sector, phase21, phase31

    def update(i):
        i *= ratio

        thisx1, thisx2 = [x1[i], xc[i], x3[i]], [xc[i], x2[i]]
        thisy1, thisy2 = [y1[i], yc[i], y3[i]], [yc[i], y2[i]]

        line1.set_data(thisx1, thisy1)
        line2.set_data(thisx2, thisy2)
        phase21.set_data(th[i], v[i])
        phase31.set_data(om[i], v[i])

        circ.center = (xc[i], yc[i])
        ax.add_patch(circ)

        time_text.set_text(time_template.format(t[i+ratio-1]))
        sector.set_theta1(90-360*t[i+ratio-1]/Tend)
        ax.add_patch(sector)

        return line1, line2, time_text, circ, sector, phase21, phase31

    n //= 7 if save == "gif" else 1
    anim = FuncAnimation(fig, update, n // ratio, interval=10, blit=True, init_func=init, repeat_delay=3000)

    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)
    plt.tight_layout()

    if save == "save":
        anim.save('Cylinder_Fall_2.html', fps=30)
    elif save == "gif":
        anim.save('./cylinder.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        update(int(10. * n / Tend))
        fig.savefig("./cylinder.svg", format="svg", bbox_inches="tight")
        # plt.show()
    else:
        print(th0)
        plt.show()


params1 = array([a0, R, m, M, L])
params2 = array([th00, om0, x0, v0])
dcm1, dcm2 = 3, 3
fmt1, fmt2 = 1 + 1 + dcm1, 1 + 1 + dcm2
for val1 in params1:
    fmt1 = max(fmt1, countDigits(val1) + 1 + dcm1)
for val2 in params2:
    fmt2 = max(fmt2, countDigits(val2) + 1 + dcm2)

parameters = [
    r"Axe x : $x \;*\; \omega$",
    r"Axe y : $x \;*\; v$",
    r"Axe c : $\vartheta$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(Tend), "",
    r"$\alpha_{{slope}} \,\:$ = {:>{width}.{dcm}f} $\rm deg$".format(a0, width=fmt1, dcm=dcm1),
    r"$r_{{wheel}} \:\,$ = {:>{width}.{dcm}f} $\rm m$".format(R, width=fmt1, dcm=dcm1),
    r"$m_{{wheel}} $ = {:>{width}.{dcm}f} $\rm kg$".format(m, width=fmt1, dcm=dcm1),
    r"$m_{{pdlm}} \,\,$ = {:>{width}.{dcm}f} $\rm kg$".format(M, width=fmt1, dcm=dcm1),
    r"$L_{{pdlm}} \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(L, width=fmt1, dcm=dcm1),
    "", r"$C_{{wheel}}$ = {:>6.3f} $\rm N \, m$".format(C), "",
    r"$g$  = {:>5.2f} $\rm m/s^2$".format(g), "",
    r"$\vartheta$ = {:>{width}.{dcm}f} $\rm deg$".format(th00, width=fmt2, dcm=dcm2),
    r"$\omega$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om0, width=fmt2, dcm=dcm2),
    r"$x$ = {:>{width}.{dcm}f} $\rm m$".format(x0, width=fmt2, dcm=dcm2),
    r"$v$ = {:>{width}.{dcm}f} $\rm m/s$".format(v0, width=fmt2, dcm=dcm2)
]
parameters[0] = r"Axe x : $\omega$"
parameters[1] = r"Axe y : $v$"
parameters[2] = r"Axe c : $\vartheta$"

#see_path_1(2, array([om*x, v*x]), th, color='Blues', shift=(0.15, -0.15),
#           var_case=2, save="no", displayedInfo=parameters)

see_animation(save="")

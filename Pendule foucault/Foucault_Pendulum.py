import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from numpy import pi, radians, sin, cos, amin, amax
from scipy.integrate import odeint

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # [m/s²]  -  accélaration de pesanteur
L = 20  # [m]     -  longueur du pendule
h = L + 1  # [m]     -  hauteur de l'accroche du pendule
phi = 90  # [°]     -  latitude
W = 1  # [tr/j] -  vitesse angulaire terrestre

# alpha_0 = 20     # [°]     -  inclinaison pendule % verticale
# d_alpha = 0      # [°/s]   -  vitesse angulaire de alpha
# beta_0  = 90     # [°]     -  beta = 0 <==> direction nord-sud
# d_beta  =  180   # [°/s]   -  vitesse angulaire de beta

alpha_0 = 20  # [°]     -  inclinaison pendule % verticale
d_alpha = 0  # [°/s]   -  vitesse angulaire de alpha
beta_0 = 90  # [°]     -  beta = 0 <==> direction nord-sud
d_beta = 0  # [°/s]   -  vitesse angulaire de beta

dt = 0.05  # [s]     -  pas de temps
Tend = 60  # [s]     -  fin de la simulation

# fps = int(1 / dt)  # animation en temps réel
fps = 100  # animation en accéléré

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
w = 2 * pi / 86400 * W
phi = radians(phi)
U0 = radians(np.array([alpha_0, d_alpha, beta_0, d_beta]))
u0 = U0[:2]

n = int(Tend // dt)
t = np.linspace(0, Tend, n + 1)


# response = input('Nombre de Frames = {} : '.format(n))
# if response == 'stop' :
#    raise RuntimeError


def f_foucault(u, _):
    a, da, b, db = u
    sa, ca, sb, cb = sin(a), cos(a), sin(b), cos(b)
    f1 = sa * ca * db * db - g / L * sa + 2 * w * db * (sin(phi) * sa * ca + cos(phi) * sa * sa * cb)
    f2 = (-2 * ca * da * db - 2 * w * da * (sin(phi) * ca + cos(phi) * sa * cb)) / sa
    return np.array([da, f1, db, f2])


# def f_classic(u,t,w,phi,g,l):
#    th,om = u
#    return array([om,-g/l*sin(th)])
# sol2 = odeint(f_classic, u0, t, args=(w,phi,g,l))
# th = sol2[:,0] ; om = sol2[:,1]


sol1 = odeint(f_foucault, U0, t)
alpha, d_alpha = sol1[:, 0], sol1[:, 1]
beta, d_beta = sol1[:, 2], sol1[:, 3]

# plt.plot(t, beta)
# plt.show()

x = L * sin(alpha) * cos(beta)
y = L * sin(alpha) * sin(beta)
z = h - L * cos(alpha)
x_min, x_max = amin(x), amax(x)
y_min, y_max = amin(y), amax(y)
z_min, z_max = amin(z), amax(z)
print('   ====== Position initiale ======    \n   x = {:.2f} \t  y = {:.2f} \t  z = {:.2f}'.format(x[0], y[0], z[0]))


########################################################################################################

#####     ================      Création de la figure      ================      #####

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(10, 32)
M = L * sin(radians(alpha_0)) * 1.1

# ax1 = fig.add_subplot(gs[:9, :17],xlim=(-M,M), ylim=(-M,M),aspect='equal')     ; ax1.grid(ls=':')
ax1 = fig.add_subplot(gs[:9, :17], xlim=(-L, L), ylim=(-L, L), aspect='equal')
ax2 = fig.add_subplot(gs[:5, 17:], xlim=(-L, L), ylim=(0, 2 * L), aspect='equal')
# ax3 = fig.add_subplot(gs[5:, 17:],xlim=(-M,M), ylim=(0, h*1.1),aspect='equal') ; ax3.grid(ls=':')
ax3 = fig.add_subplot(gs[5:, 17:], xlim=(-L, L), ylim=(0, 2 * L), aspect='equal')
ax4 = fig.add_subplot(gs[9, :17])

axs = [ax1, ax2, ax3, ax4]
for ax in axs:
    ax.grid(ls=':')

ax1.text(0.5, 0.96, 'N', fontsize=15, fontweight='bold', ha='center', wrap=True, transform=ax1.transAxes)  # Nord
ax1.text(0.94, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Est
ax1.text(0.01, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Ouest
ax1.text(0.5, 0.01, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Sud

ax2.text(0.93, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Est
ax2.text(0.01, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Ouest

ax3.text(0.93, 0.5, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Sud
ax3.text(0.01, 0.5, 'N', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Nord

ax4.axis('off')

dic1 = {'L': (L, 'm'), 'Lat': (np.degrees(phi), '°'), r'$\Omega$': (W, 'tr/j')}
dic2 = {r'$\alpha$': (alpha_0, '°'), r'$\beta$': (beta_0, '°'), 'dt': (dt, 's')}
for idx, (key, value) in enumerate(dic1.items()):
    ax4.text((idx + 1) / (len(dic1) + 2), 0.8, '{} = {}{}'.format(key, value[0], value[1]), fontsize=10, wrap=True,
             transform=ax4.transAxes)
for idx, (key, value) in enumerate(dic2.items()):
    ax4.text((idx + 1) / (len(dic2) + 2), 0.0, '{} = {}{}'.format(key, value[0], value[1]), fontsize=10, wrap=True,
             transform=ax4.transAxes)

time_template = 'time = %.1fs'
time_text = ax4.text(0.8, 0.4, '1', fontsize=15, wrap=True, transform=ax4.transAxes)

########################################################################################################

#####     ================      Animation      ================      #####

line1, = ax1.plot([], [], 'o-', lw=2, color='C1')
line2, = ax2.plot([], [], 'o-', lw=2, color='C2')
line3, = ax3.plot([], [], 'o-', lw=2, color='C2')
line4, = ax1.plot([], [], '-', lw=1, color='grey')


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])
    time_text.set_text('')
    return line1, line2, line3, line4, time_text


def animate(i):
    start = max((i - 1500, 0))

    thisx0, thisx1, thisx2 = [0, y[i]], [0, y[i]], [0, x[i]]
    thisy0, thisy1, thisy2 = [0, -x[i]], [h, z[i]], [h, z[i]]

    line1.set_data(thisx0, thisy0)
    line2.set_data(thisx1, thisy1)
    line3.set_data(thisx2, thisy2)
    line4.set_data([y[start:i + 1]], [-x[start:i + 1]])
    time_text.set_text(time_template % (i * dt))

    return line1, line2, line3, line4, time_text


interval = dt * 10
ani = anim.FuncAnimation(fig, animate, len(t), interval=interval, blit=True, init_func=init, repeat_delay=3000)
#ani.save('Foucault_2.html',fps=fps)

plt.show()

from .simulation import *

class Cylinder(Simulation):

    REQUIRED_PARAMS = ["g", "alpha", "R", "M", "L", "m", "C", "D1", "D2"]
    REQUIRED_INITIALS = ["th", "om", "x", "dx"]

    def __init__(self, setup, params, initials):
        
        # sets the attributes from setup, params, initials
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        
        # convert angles in degrees
        self.alphad = np.degrees(self.alpha)
        self.thd = np.degrees(self.th)

        # initial conditions
        self.U0 = np.array([np.pi/2 - self.th - self.alpha, self.om, self.x, self.dx])

        return
    
    def dynamics(self, t, u):
        g, a = self.g, self.alpha
        R, M, L, m = self.R, self.M, self.L, self.m
        C, D1, D2 = self.C, self.D1, self.D2

        dv = (m + M) * g * sin(a) + C / R + M * L * u[1] * u[1] * cos(u[0]) - D1*u[3] - D2/R * u[1] + \
            M * sin(u[0]) * (g * cos(a + u[0]) - D2 / (M*L) * u[1])
        dv /= (1.5*m + M * cos(u[0]) * cos(u[0]))
        dom = 1 / L * (g * cos(a + u[0]) + dv * sin(u[0]))

        return np.array([u[1], dom, u[3], dv])
    
    # not needed
    # def solve_ode(self):
    #     super().solve_ode()
    #     return
    
    def compute_kinematics(self):
        th, om, x, dx = self.full_series
        a = self.alpha
        phi = (x - self.x) / self.R 

        xc, yc = x * cos(a), -x * sin(a)  # position du centre du cercle
        x1, y1 = xc + self.R * cos(phi + a), yc - self.R * sin(phi + a)  # position d'un point sur cercle
        x3, y3 = xc - self.R * cos(phi + a), yc + self.R * sin(phi + a)  # position d'un point sur cercle
        x2, y2 = xc + self.L * cos(th + a), yc - self.L * sin(th + a)  # position du pendule

        vx2 = +dx * cos(a) - self.L * sin(th + a) * om
        vy2 = -dx * sin(a) - self.L * cos(th + a) * om
        v2 = np.hypot(vx2, vy2)

        return np.c_[xc, yc, x1, y1, x2, y2, x3, y3, vx2, vy2, v2].T

    def animate(self, fig_size=(13., 8.), save=""):
        
        th_full, om_full, x_full, dx_full = self.full_series
        th, om, x, v = self.series
        xc, yc, x1, y1, x2, y2, x3, y3, vx2, vy2, v2 = self.kinematics

        xmin, xmax = np.amin(xc) - 4.0 * max(self.R, self.L), np.amax(xc) + 4.0 * max(self.R, self.L)
        ymin, ymax = np.amin(yc) - 1.5 * max(self.R, self.L), np.amax(yc) + 1.5 * max(self.R, self.L)
        L_X, L_Y = xmax - xmin, ymax - ymin

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(211)
        tmp, = ax.plot([xmin, xmax], [ymin, ymax], "-o")
        ax.set_aspect('equal', "datalim")
        tmp.remove()

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

        xd = np.linspace(xmin, xmax, 2)
        ax.plot(xd, yc[0] - self.R / cos(self.alpha) - np.tan(self.alpha) * (xd - xc[0]), 'C0')

        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.45, 0.88, '', fontsize=ftSz2, transform=ax.transAxes)
        sector = patches.Wedge((xmin + L_X * 0.03, ymax - 0.03 * L_X),
                            L_X * 0.02, theta1=90, theta2=90, color='lightgrey')

        circ = patches.Circle((xc[0], yc[0]), radius=self.R, edgecolor=None, facecolor='lightgrey', lw=4)

        kwargs = dict(fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.06, r'$\alpha  = {:.2f} \: \rm [^\circ] $'.format(self.alphad), **kwargs)
        ax.text(0.02, 0.14, r'$\tau  = {:.2f} \: \rm [N \cdot m]$'.format(self.C), **kwargs)
        ax.text(0.17, 0.06, r'$M  = {:.2f} \: \rm [kg]$'.format(self.M), **kwargs)
        ax.text(0.17, 0.14, r'$m  = {:.2f} \: \rm [kg]$'.format(self.m), **kwargs)
        ax.text(0.33, 0.06, r'$L  = {:.2f} \: \rm [m]$'.format(self.L), **kwargs)
        ax.text(0.33, 0.14, r'$R  = {:.2f} \: \rm [m]$'.format(self.R), **kwargs)
        ax.text(0.72, 0.92, r'$\theta_0  = {:.2f} $'.format(self.thd), **kwargs)
        ax.text(0.72, 0.84, r'$\omega_0  = {:.2f} $'.format(self.om), **kwargs)
        ax.text(0.82, 0.92, r'$x_0  = {:.2f} $'.format(self.x), **kwargs)
        ax.text(0.82, 0.84, r'$v_0  = {:.2f} $'.format(self.dx), **kwargs)

        ax2.plot(th_full, dx_full, color='C1')
        #ax2.plot(om_full, v_full, color='C1')
        ax3.plot(om_full, dx_full, color='C2')

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
            thisx1, thisx2 = [x1[i], xc[i], x3[i]], [xc[i], x2[i]]
            thisy1, thisy2 = [y1[i], yc[i], y3[i]], [yc[i], y2[i]]

            line1.set_data(thisx1, thisy1)
            line2.set_data(thisx2, thisy2)
            phase21.set_data(th[i], v[i])
            phase31.set_data(om[i], v[i])

            circ.center = (xc[i], yc[i])
            ax.add_patch(circ)

            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360*self.t[i] / self.t_sim)
            ax.add_patch(sector)

            return line1, line2, time_text, circ, sector, phase21, phase31

        anim = FuncAnimation(fig, update, self.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
        plt.tight_layout()

        if save == "save":
            anim.save('Cylinder_Fall_2.html', fps=30)
        elif save == "gif":
            anim.save('./cylinder.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 10.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./cylinder.svg", format="svg", bbox_inches="tight")
            # plt.show()
        else:
            plt.show()

        return

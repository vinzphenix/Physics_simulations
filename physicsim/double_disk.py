from .simulation import *

class NestedDisks(Simulation):

    REQUIRED_PARAMS = ["g", "R", "M", "a", "m", "k", "k1"]
    REQUIRED_INITIALS = ["th", "om", "x", "dx"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.c = 2 * self.M + 3 / 2 * self.m
        self.s = self.R - self.a
        self.phi1, self.phi2 = 0., 0.
        self.thd = np.degrees(self.th)
        self.U0 = np.array([self.th, self.om, self.x, self.dx])
        return
    
    def dynamics(self, t, u):
        g, R, M, a, m, c = self.g, self.R, self.M, self.a, self.m, self.c
        th, om, x, v = u

        f1 = (g * cos(th) / (R - a) + m * cos(th) * (0.5 + sin(th)) * om * om / c) / (
                1.5 - m * (0.5 + sin(th)) ** 2 / c)
        f2 = m * (R - a) * cos(th) * om * om / c + m * (R - a) * (0.5 + sin(th)) * f1 / c

        return np.array([om, f1, v, f2])


    def compute_kinematics(self):
        th, om, x, dx = self.full_series
        R, a = self.R, self.a

        phi1 = self.phi1 + (x - self.x) / R
        phi2 = self.phi2 + (R * (phi1 - self.phi1) + (a - R) * (th - self.th)) / a

        xc1 = x
        yc1 = np.zeros_like(th)  # position du centre de l'anneau
        xc2, yc2 = x + (R - a) * cos(th), -(R - a) * sin(th)  # position du centre du disque
        x1, y1 = xc1 + R * cos(phi1), yc1 - R * sin(phi1)  # position d'un point sur l'anneau
        x2, y2 = xc2 + a * cos(phi2), yc2 - a * sin(phi2)  # position d'un point sur le disque

        vx2 = dx + (a - R) * sin(th) * om - a * sin(phi2) * ((dx + (a - R) * om) / a)
        vy2 = (a - R) * cos(th) * om - a * cos(phi2) * ((dx + (a - R) * om) / a)
        v2 = np.hypot(vx2, vy2)

        return np.c_[phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2].T


    def animate(self, figsize=(13., 6.), save=""):

        th_full, om_full, x_full, dx_full = self.full_series
        th, om, x, dx = self.series
        phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2 = self.kinematics

        R, a, M, m = self.R, self.a, self.M, self.m

        xmin, xmax = np.amin(xc1) - 1.5 * R, np.amax(xc1) + 1.5 * R
        ymin, ymax = -1.5 * R, 3 * R

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(211, autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.set_aspect("equal", "datalim")
        ax2 = fig.add_subplot(223)
        ax2.grid(ls=':')
        ax.grid(ls=':')
        ax3 = fig.add_subplot(224)
        ax3.grid(ls=':')
        ax2.set_xlabel(r'$\theta \; \rm [rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$v \; \rm [m/s]$', fontsize=ftSz2)
        ax3.set_xlabel(r'$t \; \rm [s]$', fontsize=ftSz2)

        line1, = ax.plot([], [], 'o-', lw=2, color='orange')
        line2, = ax.plot([], [], 'o-', lw=2, color='grey')
        line3, = ax.plot([], [], 'o-', lw=2, color='black')
        phase21, = ax2.plot([], [], marker='o', ms=8, color='C1')
        phase31, = ax3.plot([], [], marker='o', ms=8, color='C2')
        ax.hlines(-R, xmin - 5 * R, xmax + 5 * R, color='black', linewidth=1)

        circ1 = patches.Circle((xc1[0], yc1[0]), radius=R, facecolor='None', edgecolor='black', lw=2)
        circ2 = patches.Circle((xc2[0], yc2[0]), radius=a, facecolor='lightgrey', edgecolor='None')

        fig.tight_layout()
        xmin_, xmax_ = ax.get_xlim()
        L_X = xmax_ - xmin_
        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.45, 0.9, '', fontsize=ftSz2, transform=ax.transAxes)
        s_center = (xmax_ - L_X * 0.04, ymax - 0.04 * L_X)
        sector = patches.Wedge(s_center, L_X * 0.025, theta1=90, theta2=90, color='lightgrey')

        kwargs = dict(fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.92, r'$R  = {:.2f} $'.format(self.R), **kwargs)
        ax.text(0.02, 0.84, r'$r  = {:.2f} $'.format(self.a), **kwargs)
        ax.text(0.10, 0.92, r'$M  = {:.2f} $'.format(self.M), **kwargs)
        ax.text(0.10, 0.84, r'$m  = {:.2f} $'.format(self.m), **kwargs)
        ax.text(0.70, 0.92, r'$\theta_0  = {:.2f} $'.format(self.thd), **kwargs)
        ax.text(0.70, 0.84, r'$\omega_0  = {:.2f} $'.format(self.om), **kwargs)
        ax.text(0.80, 0.92, r'$x_0  = {:.2f} $'.format(self.x), **kwargs)
        ax.text(0.80, 0.84, r'$v_0  = {:.2f} $'.format(self.dx), **kwargs)

        ax2.plot(th_full, dx_full, color='C1')
        ax3.plot(self.full_t, dx_full, color='C2')

        #####     ================      Animation      ================      #####

        def init():
            circ1.center = (xc1[0], yc1[0])
            circ2.center = (xc2[0], yc2[0])
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            phase21.set_data([], [])
            phase31.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            return line1, line2, line3, phase21, phase31, time_text, circ1, circ2, sector

        def update(i):
            thisx0, thisx1, thisx2 = [xc1[i], xc2[i]], [xc1[i], x1[i]], [xc2[i], x2[i]]
            thisy0, thisy1, thisy2 = [yc1[i], yc2[i]], [yc1[i], y1[i]], [yc2[i], y2[i]]
            circ1.center = (xc1[i], yc1[i])
            circ2.center = (xc2[i], yc2[i])
            ax.add_patch(circ1)
            ax.add_patch(circ2)
            line1.set_data(thisx0, thisy0)
            line2.set_data(thisx1, thisy1)
            line3.set_data(thisx2, thisy2)
            phase21.set_data(th[i], dx[i])
            phase31.set_data(self.t[i], dx[i])
            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)

            return line1, line2, line3, phase21, phase31, time_text, circ1, circ2, sector

        anim = FuncAnimation(fig, update, self.n_frames+1, interval=33, blit=True, init_func=init, repeat_delay=3000)
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

        if save == "save":
            anim.save('double_disk_3.html', fps=30)
        elif save == "gif":
            anim.save('./disks.gif', writer=PillowWriter(fps=self.fps))
        elif save == "snapshot":
            t_wanted = 6.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./disks.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()
        
        return

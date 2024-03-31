from .simulation import *

class PendulumElastic(Simulation):

    REQUIRED_PARAMS = ["g", "l", "m", "k", "D"]
    REQUIRED_INITIALS = ["th", "om", "r", "dr"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.thd, self.omd = np.degrees(self.th), np.degrees(self.om)
        self.r_eq = self.m * self.g / self.k
        self.U0 = np.array([self.th, self.om, self.r, self.dr])
        return
    
    def dynamics(self, t, u):    
        g, l, m, k, D = self.g, self.l, self.m, self.k, self.D
        th, om, r, v = u
        d2th = -2 * v * om / (r + l) - g * sin(th) / (r + l) - D / m * (r + l) * om
        d2r = (r + l) * om * om + g * cos(th) - k / m * r - D / m * v
        return np.array([om, d2th, v, d2r])

    def compute_kinematics(self):

        th, om, r, dr = self.full_series
        g, l, m, k = self.g, self.l, self.m, self.k

        x =   (r + l) * sin(th)
        y = - (r + l) * cos(th)

        vx =   dr * sin(th) + (r + l) * cos(th) * om
        vy = - dr * cos(th) + (r + l) * sin(th) * om
        speed = np.hypot(vx, vy)

        _, d2th, _, d2r = self.dynamics(self.full_t, [th, om, r, dr])
        # d2r = (r + l) * om * om + g * cos(th) - k / m * r
        # d2th = -2 * dr * om / (r + l) - g * sin(th) / (r + l)
        ddx = + (d2r - (r + l) * om * om) * sin(th) + (2 * dr * om + (r + l) * d2th) * cos(th)
        ddy = - (d2r - (r + l) * om * om) * cos(th) + (2 * dr * om + (r + l) * d2th) * sin(th)
        acc = np.hypot(ddx, ddy)

        return np.c_[x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc].T


    def animate(self, figsize=(13., 6.), save="", phaseSpace=0):

        th_full, om_full, r_full, dr_full = self.full_series
        x_full, y_full = self.full_kinematics[:2]
        th, om, r, dr = self.series
        x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc = self.kinematics
        k = self.oversample

        max_x = 1.1 * np.amax(r + self.l)
        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs
        ax.grid(ls=':')
        ax2.grid(ls=':')
        tmp, = ax.plot([-max_x, max_x], [-max_x, max_x], 'k-', lw=1)
        ax.set_aspect('equal', 'datalim')
        tmp.remove()

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], 'o-', lw=4, color='grey', alpha=0.3)
        line3, = ax.plot([], [], '-', lw=1, color='grey', alpha=0.8)

        phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C1')

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xm, delta_x = 0.5*(xmin + xmax), xmax - xmin
        ym, delta_y = 0.5*(ymin + ymax), ymax - ymin
        delta = min(delta_x, delta_y)
        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.79, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
        s_center = (xmax - 0.1 * delta, ymin + 0.1*delta)
        sector = patches.Wedge(s_center, 0.04*delta, theta1=90, theta2=90, color='lightgrey')

        kw = dict(transform=ax.transAxes, fontsize=ftSz3, wrap=True)
        ax.text(0.02, 0.86, r'$k  \,\, = {:.2f} \,N/m$'.format(self.k), **kw)
        ax.text(0.02, 0.82, r'$\ell \:\; = {:.2f} \,m$'.format(self.l), **kw)
        ax.text(0.02, 0.78, r'$m  = {:.2f} \,kg$'.format(self.m), **kw)

        ax.text(0.02, 0.96, r'$r         = {:.2f} $'.format(self.r), **kw)
        ax.text(0.02, 0.92, r'$\dot r    = {:.2f} $'.format(self.dr), **kw)
        ax.text(0.17, 0.96, r'$\vartheta = {:.2f} $'.format(self.thd), **kw)
        ax.text(0.17, 0.92, r'$\omega    = {:.2f} $'.format(self.om), **kw)
        # 87 90 93 96
        if phaseSpace == 0:
            ax2.plot(th_full, om_full, color='C0', label=r'$\vartheta \; :\; \omega$')
            ax2.plot(r_full - self.r_eq, dr_full, color='C1', label=r'$r \;\, : \; \dot r$')
            ax2.legend(fontsize=ftSz3)
        elif phaseSpace == 1:
            ax2.plot(x_full, y_full, color='C0', label='Trajectoire')
            ax2.set_aspect('equal')
        else:
            ax2.plot(th_full, r_full - self.r_eq, color='C1', label=r'$\vartheta / r$')
            ax2.plot(th_full, dr_full, color='C2', label=r'$\vartheta / \dot r$')
            ax2.plot(om_full, r_full - self.r_eq, color='C3', label=r'$\omega / r$')
            ax2.plot(om_full, dr_full, color='C4', label=r'$\omega / \dot r$')
            ax2.plot(r_full, dr_full, color='C5', label=r'$r / \dot r$')
            ax2.legend(fontsize=ftSz3)

        #####     ================      Animation      ================      #####

        def init():

            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            phase1.set_data([], [])
            phase2.set_data([], [])

            time_text.set_text('')
            sector.set_theta1(90)

            liste = [line1, line2, line3, phase1, phase2, time_text, sector]

            return tuple(liste)

        def update(i):
            start = max(0, i - 10000)
            line1.set_data([0, x[i]], [0, y[i]])
            line2.set_data(
                [0, (self.l + self.r_eq) * sin(th[i])], 
                [0, -(self.l + self.r_eq) * cos(th[i])]
            )
            line3.set_data([x_full[k*start:k*i + 1]], [y_full[k*start:k*i + 1]])

            if phaseSpace == 0:
                phase1.set_data(th[i], om[i])
                phase2.set_data(r[i] - self.r_eq, dr[i])
            elif phaseSpace == 1:
                phase1.set_data(x[i], y[i])
            else:
                phase1.set_data(th[i], r[i] - self.r_eq)
                phase2.set_data(om[i], dr[i])

            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)

            liste = [line1, line2, line3, phase1, phase2, time_text, sector]
            return tuple(liste)

        anim = FuncAnimation(
            fig, update, self.n_frames + 1, interval=10, 
            blit=True, init_func=init, repeat_delay=3000
        )
        fig.tight_layout()

        if save == "save":
            anim.save('Pendule_Elastique_1', fps=30)
        elif save == "gif":
            anim.save('./elastic_pendulum.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 8.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./elastic_pendulum.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

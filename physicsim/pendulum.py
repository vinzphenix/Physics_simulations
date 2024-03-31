from .simulation import *

class Pendulum(Simulation):
    
    REQUIRED_PARAMS = ["g", "D", "m", "l"]
    REQUIRED_INITIALS = ["phi", "om"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.phid = np.degrees(self.phi)
        self.U0 = np.array([self.phi, self.om])
        return

    def dynamics(self, t, u):
        g, D, m, l = self.g, self.D, self.m, self.l
        phi, om = u
        return np.array([om, - g / l * sin(phi) - D / m * om])

    def compute_kinematics(self):
        phi, om = self.full_series
        x, vx = self.l * sin(phi), self.l * om * cos(phi)
        y, vy = -self.l * cos(phi), self.l * om * sin(phi)
        speed = np.hypot(vx, vy)
        return np.c_[x, y, speed].T


    def animate(self, figsize=(13., 6.), save=""):

        phi_full, om_full = self.full_series
        phi, om = self.series
        x, y, speed = self.kinematics
        x_full, y_full, _ = self.full_kinematics
        k = self.oversample

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        ax = axs[0]
        tmp, = ax.plot([-self.l * 1.15, self.l * 1.15], [-1.1 * self.l, 1.2 * self.l])
        ax.set_aspect("equal")
        tmp.remove()
        ax.grid(ls=':')

        ax2 = axs[1]
        ax2.grid(ls=':')
        ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], '-', lw=1, color='grey')
        phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.40, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
        s_center = (1 * self.l, -0.95 * self.l)
        sector = patches.Wedge(s_center, self.l / 10, theta1=90, theta2=90, color='lightgrey')

        kwargs = dict(fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.05, 0.95, r'$L  = {:.2f} \; m$'.format(self.l), **kwargs)
        ax.text(0.05, 0.90, r'$m  = {:.2f} \; kg$'.format(self.m), **kwargs)
        ax.text(0.75, 0.95, r'$\varphi_1  = {:.2f} $'.format(self.phid), **kwargs)
        ax.text(0.75, 0.90, r'$\omega_1  = {:.2f} $'.format(self.om), **kwargs)
        ax2.plot(phi_full, om_full, color='C1', label='pendule inertie')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            phase1.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            return line1, line2, phase1, time_text, sector

        def update(i):
            start = max(0, i - 100)
            thisx = [0, x[i]]
            thisy = [0, y[i]]
            line1.set_data(thisx, thisy)
            line2.set_data([x_full[k*start:k*i + 1]], [y_full[k*start:k*i + 1]])
            phase1.set_data(phi[i], om[i])
            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)

            return line1, line2, phase1, time_text, sector

        anim = FuncAnimation(fig, update, self.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
        fig.tight_layout()

        if save == "save":
            anim.save('Pendule_simple_1', fps=30)
        elif save == "gif":
            anim.save('./simple_pendulum.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 4.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./simple_pendulum.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

        return

from .simulation import *

class VerticalPendulum(Simulation):

    REQUIRED_PARAMS = ["g", "l", "mp", "mb", "F", "w"]
    REQUIRED_INITIALS = ["x", "dx", "phi", "om"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.M = self.mp + self.mb
        self.phid = np.degrees(self.phi)
        self.U0 = np.array([self.x, self.dx, self.phi, self.om])
        return
    
    def dynamics(self, t, u):
        x, dx, phi, om = u
        s, c = sin(phi), cos(phi)
        g, mp, mb, l = self.g, self.mp, self.mb, self.l
        force = self.F * cos(self.w * t)
        f1 = (-g * mp * s * s + force - mp * l * om * om * c) / (mb + mp * c * c)
        f2 = s * (g - f1) / l
        return np.array([dx, f1, om, f2])


    def compute_kinematics(self):
        x, dx, phi, om = self.full_series
        xb, yb = np.zeros_like(x), -x
        xp, yp = -self.l * sin(phi), -x + self.l * cos(phi)
        vxp = -self.l * om * cos(phi)
        vyp = -dx - self.l * om * sin(phi)
        vp = np.hypot(vxp, vyp)
        return np.c_[xb, yb, xp, yp, vp].T


    def animate(self, figsize=(13., 6.), save=""):

        x_full, dx_full, phi_full, om_full = self.full_series
        xp_full, yp_full = self.full_kinematics[2:4]
        x, dx, phi, om = self.series
        xb, yb, xp, yp, vp = self.kinematics
        
        k = self.oversample
        y_min, y_max = np.amin(yb), np.amax(yb)
        d = abs((y_min - 1.1 * self.l) - (y_max + 1.1 * self.l))

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs
        ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)
        ax.grid(ls=':')
        ax2.grid(ls=':')
        
        tmp, = ax.plot([-1.1*self.l, 1.1*self.l], [y_min - 1.1 * self.l, y_max + 1.1 * self.l])
        ax.set_aspect("equal", "datalim")
        tmp.remove()

        line1, = ax.plot([], [], 'o-', lw=2, color='C2')
        line2, = ax.plot([], [], '-', lw=1, color='grey')
        rect = plt.Rectangle((-self.l, yb[0] - 0.1 * self.l), 2 * self.l, 0.2 * self.l, color='C1')
        ax.add_patch(rect)

        phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.48, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
        s_center = (d / 2 * 0.85, (y_min - 1.2 * self.l) * 0.85)
        sector = patches.Wedge(s_center, d / 20, theta1=90, theta2=90, color='lightgrey')

        kw = dict(fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.96, r'$l  \: \: \: = {:.3f} \: kg$'.format(self.l), **kw)
        ax.text(0.02, 0.92, r'$m  = {:.3f} \: kg$'.format(self.mp), **kw)
        ax.text(0.02, 0.88, r'$M  = {:.3f} \: kg$'.format(self.mb), **kw)
        ax.text(0.21, 0.96, r'$F   = {:.2f} \: N$'.format(self.F), **kw)
        ax.text(0.21, 0.92, r'$\omega  = {:.2f} \: rad/s$'.format(self.w), **kw)

        ax.text(0.74, 0.96, r'$x = {:.2f} $'.format(self.x), **kw)
        ax.text(0.74, 0.92, r'$v = {:.2f} $'.format(self.dx), **kw)
        ax.text(0.86, 0.96, r'$\varphi  = {:.2f} $'.format(self.phid), **kw)
        ax.text(0.86, 0.92, r'$\dot{{\varphi}}  = {:.2f} $'.format(self.om), **kw)

        ax2.plot(phi_full, om_full, color='C1')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            rect.set_bounds(-self.l, yb[0] - 0.1 * self.l, 2 * self.l, 0.2 * self.l)
            phase1.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            return line1, line2, rect, phase1, time_text, sector

        def update(i):
            start = max(0, i-50000)
            thisx = [xb[i], xp[i]]
            thisy = [yb[i], yp[i]]

            line1.set_data(thisx, thisy)
            line2.set_data([xp_full[k*start:k*i + 1]], [yp_full[k*start:k*i + 1]])
            rect.set_y(yb[i] - 0.1 * self.l)

            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)

            phase1.set_data(phi[i], om[i])

            return line1, line2, rect, phase1, time_text, sector

        anim = FuncAnimation(
            fig, update, self.n_frames+1, interval=20, 
            repeat_delay=3000, init_func=init, blit=True
        )
        fig.tight_layout()

        if save == "save":
            anim.save('Vertical_Inverted_Pendulum_3.html', fps=30)
        elif save == "gif":
            anim.save('./pendulum_vertical.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 15.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./pendulum_vertical.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

        return
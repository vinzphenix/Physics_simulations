from .simulation import *

class FoucaultPendulum(Simulation):

    REQUIRED_PARAMS = ["g", "L", "phi", "W"]
    REQUIRED_INITIALS = ["alpha", "dalpha", "beta", "dbeta"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.h = self.L + 1
        self.w = 2 * np.pi / 86400 * self.W
        
        self.phid, self.alphad, self.betad = np.degrees([self.phi, self.alpha, self.beta])
        self.U0 = np.array([self.alpha, self.dalpha, self.beta, self.dbeta])
        return
    
    def dynamics(self, t, u):
        g, L, phi, w = self.g, self.L, self.phi, self.w
        a, da, b, db = u
        sa, ca, sb, cb = sin(a), cos(a), sin(b), cos(b)
        f1 = sa * ca * db * db - g / L * sa + 2 * w * db * (sin(phi) * sa * ca + cos(phi) * sa * sa * cb)
        f2 = (-2 * ca * da * db - 2 * w * da * (sin(phi) * ca + cos(phi) * sa * cb)) / sa
        return np.array([da, f1, db, f2])


    def compute_kinematics(self):
        alpha, dalpha, beta, dbeta = self.full_series
        L, h = self.L, self.h
        x = L * sin(alpha) * cos(beta)
        y = L * sin(alpha) * sin(beta)
        z = h - L * cos(alpha)
        return np.c_[x, y, z].T


    def animate(self, figsize=(12., 7.), save=""):

        a_full, da_full, b_full, db_full = self.full_series
        x_full, y_full, z_full = self.full_kinematics
        a, da, b, db = self.series
        x, y, z = self.kinematics

        k = self.oversample
        x_min, x_max = np.amin(x), np.amax(x)
        y_min, y_max = np.amin(y), np.amax(y)
        z_min, z_max = np.amin(z), np.amax(z)

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        
        fig = plt.figure(figsize=figsize)
        M = self.L * np.amax(np.sin(self.alpha)) * 1.1
        ax1 = fig.add_subplot(121, xlim=(-M, M), ylim=(-M, M))
        ax2 = fig.add_subplot(243, xlim=(-M, M), ylim=(-0.1*M, 1.9*M))
        ax3 = fig.add_subplot(247, xlim=(-M, M), ylim=(-0.1*M, 1.9*M), sharex=ax2)
        all_bounds = [[-M, M, -M, M], [-M, M, -0.1*M, 1.9*M], [-M, M, -0.1*M, 1.9*M]]
        for ax, bounds in zip([ax1, ax2, ax3], all_bounds):
            tmp, = ax1.plot([-M, M], [-M, M])
            ax.set_aspect("equal", "datalim")
            tmp.remove()

        ax_a = fig.add_subplot(244)
        ax_b = fig.add_subplot(248, sharex=ax_a)
        # ax_b.set_xlabel(r"$t$", fontsize=ftSz2)
        # ax_a.set_ylabel(r"$\alpha$", fontsize=ftSz2)
        # ax_b.set_ylabel(r"$\beta$", fontsize=ftSz2)
        # if save != "gif":
        #     ax_a.xaxis.set_ticklabels([])
        #     ax_b.xaxis.set_ticklabels([])

        axs = [ax1, ax2, ax3, ax_a, ax_b]
        for ax in axs:
            ax.grid(ls=':')

        kw = dict(fontweight='bold', wrap=True, transform=ax1.transAxes)
        ax1.text(0.5, 0.96, 'N', fontsize=15, ha='center', **kw)  # Nord
        ax1.text(0.96, 0.5, 'E', fontsize=15, **kw)  # Est
        ax1.text(0.01, 0.5, 'O', fontsize=15, **kw)  # Ouest
        ax1.text(0.5, 0.01, 'S', fontsize=15, **kw)  # Sud

        kw = dict(fontweight='bold', wrap=True, transform=ax2.transAxes)
        ax2.text(0.93, 0.5, 'E', fontsize=15, **kw)  # Est
        ax2.text(0.03, 0.5, 'O', fontsize=15, **kw)  # Ouest

        kw = dict(fontweight='bold', wrap=True, transform=ax3.transAxes)
        ax3.text(0.93, 0.5, 'S', fontsize=15, **kw)  # Sud
        ax3.text(0.03, 0.5, 'N', fontsize=15, **kw)  # Nord

        kw = dict(fontsize=ftSz3, transform=ax1.transAxes)
        ax1.text(0.05, 0.92, r'$\Omega = {:.0f} \; tr/j$'.format(self.W), **kw)
        ax1.text(0.05, 0.87, r'$L = {:.0f} \; m$'.format(self.L), **kw)
        ax1.text(0.05, 0.82, r'$\phi = {:.0f} \; °$'.format(self.phid), **kw)
        ax1.text(0.70, 0.87, r'$\alpha_0 = {:.2f} \; °$'.format(self.alphad), **kw)
        ax1.text(0.70, 0.82, r'$\beta_0 = {:.2f} \; °$'.format(self.betad), **kw)

        time_template = r'$t \; = {:.2f} \; s$' if save == "snapshot" else r'$t \;\:\: = \mathtt{{{:.2f}}} \; s$'
        time_text = ax1.text(0.70, 0.92, "", **kw)

        beta_wrapped = np.remainder(b_full + np.pi, 2 * np.pi) - np.pi
        positions = np.where(np.abs(np.diff(beta_wrapped)) > np.pi)[0] + 1
        t_with_nan = np.insert(self.full_t, positions, np.nan)
        beta_with_nan = np.insert(beta_wrapped, positions, np.nan)
        period = self.full_t[positions[0]] if len(positions) > 0 else self.t_sim

        ax_a.plot(self.full_t, a_full, color='C0')
        ax_b.plot(t_with_nan, beta_with_nan, color='C1')

        line1, = ax1.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax2.plot([], [], 'o-', lw=2, color='C2')
        line3, = ax3.plot([], [], 'o-', lw=2, color='C2')
        line4, = ax1.plot([], [], '-', lw=1, color='grey')
        cursor_a, = ax_a.plot([], [], 'o', markersize=5, color='C0')
        cursor_b, = ax_b.plot([], [], 'o', markersize=5, color='C1')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            line4.set_data([], [])
            time_text.set_text('')
            cursor_a.set_data([], [])
            cursor_b.set_data([], [])
            return line1, line2, line3, line4, time_text, cursor_a, cursor_b, #ax_a, ax_b

        def update(i):
            start = max((i - 1500, 0))
            thisx0, thisx1, thisx2 = [0, y[i]], [0, y[i]], [0, x[i]]
            thisy0, thisy1, thisy2 = [0, -x[i]], [self.h, z[i]], [self.h, z[i]]

            line1.set_data(thisx0, thisy0)
            line2.set_data(thisx1, thisy1)
            line3.set_data(thisx2, thisy2)
            line4.set_data([y_full[k*start:k*i + 1]], [-x_full[k*start:k*i + 1]])
            cursor_a.set_data(self.t[i], a_full[k*i])
            cursor_b.set_data(self.t[i], beta_wrapped[k*i])
            time_text.set_text(time_template.format(self.t[i]))
            # start = max(0., t[i] - period / 2.)
            # for ax_ in [ax_a, ax_b]:
            #     ax_.set_xlim([start, start + period])

            return line1, line2, line3, line4, time_text, cursor_a, cursor_b, #ax_a, ax_b

        anim = FuncAnimation(fig, update, self.n_frames+1, interval=5, blit=(save != "gif"), init_func=init, repeat_delay=3000)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.18, hspace=0.1, top=0.98, bottom=0.09)

        if save == "save":
            anim.save('Foucault_2.html', fps=self.fps)
        elif save == "gif":
            anim.save('./foucault_pendulum.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 20.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./foucault_pendulum.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

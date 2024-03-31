from .simulation import *
# ATTENTION : convention angles relatifs


class DrivenPendulum(Simulation):

    REQUIRED_PARAMS = ["g", "l1", "l2"]
    REQUIRED_INITIALS = ["phi1", "om1", "phi2", "om2"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.phi1d = np.degrees(initials['phi1'])
        self.phi2d = np.degrees(initials['phi2'])
        self.U0 = np.array([self.phi2, self.om2 - self.om1])
        return
    
    def dynamics(self, t, u):
        l1, l2, g = self.l1, self.l2, self.g
        phi2, om2 = u
        phi1_phi2 = (self.phi1 + self.om1 * t) + phi2
        f1 = -l1 * self.om1 * self.om1 / l2 * sin(phi2) - g / l2 * sin(phi1_phi2)
        return np.array([om2, f1])


    def solve_ode(self):

        start = perf_counter()
        sol = odeint(self.dynamics, self.U0, self.full_t, tfirst=True)
        end = perf_counter()
        print(f"\tElapsed time : {end-start:.3f} seconds")
        
        phi1 = self.phi1 + self.om1 * self.full_t
        om1 = np.ones_like(self.full_t) * self.om1
        
        self.full_series = np.c_[phi1, om1, sol].T
        self.series = self.full_series[:, ::self.oversample]        
        
        self.full_kinematics = self.compute_kinematics()
        self.kinematics = self.full_kinematics[:, ::self.oversample]
        
        return

    def compute_kinematics(self):

        l1, l2, g = self.l1, self.l2, self.g
        phi1, om1, phi2, om2 = self.full_series
        phi12, om12 = phi1 + phi2, om1 + om2

        x1, y1 = l1 * sin(phi1), -l1 * cos(phi1)
        x2, y2 = x1 + l2 * sin(phi12), y1 - l2 * cos(phi12)

        vx2 = l1 * cos(phi1) * om1 + l2 * cos(phi12) * (om12)
        vy2 = l1 * sin(phi1) * om1 + l2 * sin(phi12) * (om12)
        v2 = np.hypot(vx2, vy2)

        acx2 = -l1 * sin(phi1) * om1**2 - l2 * sin(phi12) * (om12)**2 \
            + l2 * cos(phi12) * (
                -l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(phi12)
            )
        acy2 = +l1 * cos(phi1) * om1**2 + l2 * cos(phi12) * (om12)**2 \
            + l2 * sin(phi12) * (
                -l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(phi12)
            )
        ac2 = np.hypot(acx2, acy2)

        T = g * cos(phi1) + (l1 * om1 * om1 * cos(phi2) + l2 * (om12) ** 2)

        return np.c_[x1, y1, x2, y2, v2, ac2, T].T


    def animate(self, figsize=(13., 6.), save=False):
        
        phi1_full, om1_full, phi2_full, om2_full = self.full_series
        x2_full, y2_full = self.full_kinematics[2:4]
        phi1, om1, phi2, om2 = self.series
        x1, y1, x2, y2, v2, ac2, T = self.kinematics
        k = self.oversample

        L, T_max = self.l1 + self.l2, np.amax(T)

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs

        # To make matplotlib understand the aspect ratio
        tmp, = ax.plot([1.1*L, 1.1*L, -1.1*L, -1.1*L], [1.1*L, -1.1*L, -1.1*L, 1.1*L], ls='--')
        ax.set_aspect('equal', adjustable='datalim')
        tmp.remove()
        ax2.set_xlabel(r'$\varphi \; \rm [rad]$', fontsize=12)
        ax2.set_ylabel(r'$\omega \; \rm [rad/s]$', fontsize=12)
        ax.grid(ls=':')
        ax2.grid(ls=':')

        time_template = r'$t = %.2f \rm s$'
        time_text = ax.text(0.38, 0.94, '1', fontsize=15, wrap=True, transform=ax.transAxes)
        sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

        fontsize = 11
        kw = dict(fontsize=fontsize, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.96, r'$l_1 = {:.2f} \: \rm m$'.format(self.l1), **kw)
        ax.text(0.02, 0.92, r'$l_2 = {:.2f} \: \rm m$'.format(self.l2), **kw)

        ax.text(0.56, 0.96, r'$\varphi_1  = {:.2f} \;\rm deg$'.format(self.phi1d), **kw)
        ax.text(0.56, 0.92, r'$\varphi_2  = {:.2f} \;\rm deg$'.format(self.phi2d), **kw)
        ax.text(0.76, 0.96, r'$\omega_1  = {:.2f} \;\rm rad/s$'.format(self.om1), **kw)
        ax.text(0.76, 0.92, r'$\omega_2  = {:.2f} \;\rm rad/s$'.format(self.om2), **kw)

        ax2.plot(phi1_full + phi2_full, om1_full + om2_full, color='C2')

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], 'o-', lw=2, color='C2')
        line3, = ax.plot([], [], '-', lw=1, color='grey')
        rect = plt.Rectangle((L * 1.0, 0), L * 0.05, T[0] / T_max * L)
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')

        #####     ================      Animation      ================      #####

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            time_text.set_text('')
            rect.set_bounds(L * 1.05, 0, L * 0.05, T[0] / T_max * L)
            phase2.set_data([], [])
            sector.set_theta1(90)
            return line1, line2, line3, time_text, rect, phase2, sector

        def update(i):
            start = max((i - 50000, 0))

            thisx1, thisx2 = [0, x1[i]], [x1[i], x2[i]]
            thisy1, thisy2 = [0, y1[i]], [y1[i], y2[i]]

            line1.set_data(thisx1, thisy1)
            line2.set_data(thisx2, thisy2)
            line3.set_data(x2_full[k*start:k*i + 1], y2_full[k*start:k*i + 1])

            time_text.set_text(time_template % (self.t[i]))

            rect.set_bounds(L * 1.0, 0, L * 0.05, T[i] / T_max * L)
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(rect)
            ax.add_patch(sector)

            phase2.set_data(phi1[i] + phi2[i], om1[i] + om2[i])

            return line1, line2, line3, time_text, rect, phase2, sector

        anim = FuncAnimation(fig, update, self.n_frames + 1, interval=20, blit=True, init_func=init, repeat_delay=5000)
        fig.tight_layout()

        if save == "save":
            anim.save('Pendule_entraine_4.html', fps=30)
        elif save == "gif":
            anim.save('./driven_pendulum.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 20.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./driven_pendulum.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()
        
        return

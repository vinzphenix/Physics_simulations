from .simulation import *

# Angles are now absolute
class DrivenPendulum(Simulation):

    REQUIRED_PARAMS = ["g", "l1", "l2"]
    REQUIRED_INITIALS = ["phi1", "om1", "phi2", "om2"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.phi1d = np.degrees(initials['phi1'])
        self.phi2d = np.degrees(initials['phi2'])
        self.ref_time = np.sqrt(self.l2 / self.g)
        self.lbd = self.l1 / self.l2
        self.phi1_t = lambda t: self.phi1 + self.om1 * self.ref_time * t  # dimensionless
        self.om1_t = lambda t: 0.*t + self.om1 * self.ref_time            # dimensionless
        self.U0 = np.array([self.phi2, self.om2 * self.ref_time])  # dimensionless
        return

    def dynamics(self, t, u):  # dimensionless
        phi1, om1 = self.phi1_t(t), self.om1_t(t)
        phi2, om2 = u
        dom2 = self.lbd * om1*om1 * sin(phi1 - phi2) - sin(phi2)
        return np.array([om2, dom2])

    def solve_ode(self):

        start = perf_counter()
        sol = odeint(self.dynamics, self.U0, self.full_t / self.ref_time, tfirst=True)
        end = perf_counter()
        print(f"\tElapsed time : {end-start:.3f} seconds")
        
        phi1 = self.phi1_t(self.full_t / self.ref_time)
        om1 = self.om1_t(self.full_t) / self.ref_time
        phi2, om2 = sol.T
        om2 /= self.ref_time
        
        self.full_series = np.c_[phi1, om1, phi2, om2].T
        self.series = self.full_series[:, ::self.oversample]
        
        self.full_kinematics = self.compute_kinematics()
        self.kinematics = self.full_kinematics[:, ::self.oversample]
        
        return

    def compute_kinematics(self):

        l1, l2, g = self.l1, self.l2, self.g
        phi1, om1, phi2, om2 = self.full_series
        c12, s12 = cos(phi1 - phi2), sin(phi1 - phi2)

        x1, y1 = l1 * sin(phi1), -l1 * cos(phi1)
        x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

        vx2 = l1 * cos(phi1) * om1 + l2 * cos(phi2) * om2
        vy2 = l1 * sin(phi1) * om1 + l2 * sin(phi2) * om2
        v2 = np.hypot(vx2, vy2)

        _, dom2 = self.dynamics(self.full_t, [phi2, om2])
        dom2 /= self.ref_time ** 2

        acx2 = -l1 * sin(phi1) * om1**2 - l2 * sin(phi2) * om2**2 + l2 * cos(phi2) * dom2
        acy2 = +l1 * cos(phi1) * om1**2 + l2 * cos(phi2) * om2**2 + l2 * sin(phi2) * dom2
        ac2 = np.hypot(acx2, acy2)

        # Tension in each rod (assumption that mass is concentrated at the end of each rod)
        m1, m2 = 1., 1.
        T1 = (m1 + m2) * (l1 * om1 * om1 + g * cos(phi1)) + m2 * l2 * (om2 * om2 * c12 - dom2 * s12)
        T2 = m2 * l1 * om1 * om1 * c12 + m2 * (l2 * om2 * om2 + g * cos(phi2))
        # resultant = T1*T1 + T2*T2 - 2*T1*T2 * c12

        return np.c_[x1, y1, x2, y2, v2, ac2, T1, T2].T


    def animate(self, figsize=(13., 6.), wrap=False, save=False):
        
        phi1_full, om1_full, phi2_full, om2_full = self.full_series
        x2_full, y2_full = self.full_kinematics[2:4]
        phi1, om1, phi2, om2 = self.series
        x1, y1, x2, y2, v2, ac2, T1, T2 = self.kinematics
        k = self.oversample

        if wrap:
            _, series_plot, _ = self.get_cut_series(0, 2)
            phi1_plot, om1_plot, phi2_plot, om2_plot = series_plot
        else:
            phi1_plot, om1_plot, phi2_plot, om2_plot = self.full_series

        L = self.l1 + self.l2
        T_max = max(np.amax(np.abs(T1)), np.amax(np.abs(T2)))

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs

        # To make matplotlib understand the aspect ratio
        tmp, = ax.plot([-1.1*L, 1.2*L], [-1.1*L, 1.1*L], ls='--')
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

        ax2.plot(phi2_plot, om2_plot, color='C2')

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], 'o-', lw=2, color='C2')
        line3, = ax.plot([], [], '-', lw=1, color='grey')
        rect1 = plt.Rectangle((L * 1.050, 0), L * 0.05, T1[0] / T_max * L, color='C1', alpha=0.5)
        rect2 = plt.Rectangle((L * 1.125, 0), L * 0.05, T2[0] / T_max * L, color='C2', alpha=0.5)
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')

        #####     ================      Animation      ================      #####

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            time_text.set_text('')
            rect1.set_bounds(L * 1.025, 0, L * 0.0, T1[0] / T_max * L)
            rect2.set_bounds(L * 1.125, 0, L * 0.0, T2[0] / T_max * L)
            phase2.set_data([], [])
            sector.set_theta1(90)
            return line1, line2, line3, time_text, phase2, sector, rect1, rect2

        def update(i):
            start = max((i - 50000, 0))

            thisx1, thisx2 = [0, x1[i]], [x1[i], x2[i]]
            thisy1, thisy2 = [0, y1[i]], [y1[i], y2[i]]

            line1.set_data(thisx1, thisy1)
            line2.set_data(thisx2, thisy2)
            line3.set_data(x2_full[k*start:k*i + 1], y2_full[k*start:k*i + 1])

            time_text.set_text(time_template % (self.t[i]))

            rect1.set_bounds(L * 1.050, 0, L * 0.05, T1[i] / T_max * L)
            rect2.set_bounds(L * 1.125, 0, L * 0.05, T2[i] / T_max * L)
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(sector)
            
            phase2.set_data(phi2[i], om2[i])

            return line1, line2, line3, time_text, phase2, sector, rect1, rect2

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

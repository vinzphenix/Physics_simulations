from .simulation import *

class DoublePendulum(Simulation):
    
    REQUIRED_PARAMS = ["g", "l1", "l2", "m1", "m2"]
    REQUIRED_INITIALS = ["phi1", "om1", "phi2", "om2"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        
        if params.get('adim', False):
            self.ref_time = 1.0
        else:
            self.ref_time = np.sqrt(self.l1 / self.g)
        
        self.l = self.l2 / self.l1
        self.mu = self.m2 / (self.m1 + self.m2)
        # self.om1a, self.om2a = self.om1 * self.ref_time, self.om2 * self.ref_time
        self.phi1d, self.phi2d = np.degrees(self.phi1), np.degrees(self.phi2)
        # self.om1d, self.om2d = self.om1a * np.sqrt(self.g / self.l1), self.om2a * np.sqrt(self.g / self.l1)
        self.L = self.l1 + self.l2

        self.U0 = np.array([
            self.phi1, self.om1 * self.ref_time, 
            self.phi2, self.om2 * self.ref_time
        ])

        return

    def dynamics(self, t, u):
        # dimentionless form of the equations of motion
        l, mu = self.l, self.mu
        th1, w1, th2, w2 = u[0], u[1], u[2], u[3]
        Cos, Sin = cos(th1 - th2), sin(th1 - th2)
        f1 = (mu * Cos * (sin(th2) - w1 * w1 * Sin) - l * mu * w2 * w2 * Sin - sin(th1)) / (1. - mu * Cos * Cos)
        f2 = Sin * (cos(th1) + w1 * w1 + l * mu * w2 * w2 * Cos) / (l - l * mu * Cos * Cos)
        return np.array([w1, f1, w2, f2])
    
    def solve_ode(self, verbose=True):

        start = perf_counter()
        sol = odeint(self.dynamics, self.U0, self.full_t / self.ref_time, tfirst=True)
        end = perf_counter()
        if verbose:
            print(f"\tElapsed time : {end-start:.3f} seconds")
        
        self.full_series = sol.T
        self.full_series[[1, 3]] /= self.ref_time  # dimensionalize the angular velocities
        self.series = self.full_series[:, ::self.oversample]
        
        self.full_kinematics = self.compute_kinematics()
        self.kinematics = self.full_kinematics[:, ::self.oversample]        
        return

    def compute_kinematics(self):
        l1, l2 = self.l1, self.l2
        m1, m2 = self.m1, self.m2
        phi1, om1, phi2, om2 = self.full_series

        x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
        x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

        vx1, vy1 = l1 * om1 * cos(phi1), l1 * om1 * sin(phi1)
        vx2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
        vy2 = l1 * om1 * sin(phi1) + l2 * om2 * sin(phi2)
        v1, v2 = np.hypot(vx1, vy1), np.hypot(vx2, vy2)

        _, dom1, _, dom2 = self.dynamics(
            0., 
            np.array([phi1, om1 / self.ref_time, phi2, om2 / self.ref_time])
        )
        dom1, dom2 = dom1 / self.ref_time ** 2, dom2 / self.ref_time ** 2

        acx2 = l1 * dom1 * cos(phi1) - l1 * om1 * om1 * sin(phi1) + l2 * dom2 * cos(phi2) - l2 * om2 * om2 * sin(phi2)
        acy2 = l1 * dom1 * sin(phi1) + l1 * om1 * om1 * cos(phi1) + l2 * dom2 * sin(phi2) + l2 * om2 * om2 * cos(phi2)
        ac2 = np.hypot(acx2, acy2)

        c12, s12 = cos(phi1 - phi2), sin(phi1 - phi2)
        T1 = (m1 + m2) * (l1 * om1 * om1 + self.g * cos(phi1)) + m2 * l2 * (om2 * om2 * c12 - dom2 * s12)
        T2 = m2 * l1 * (om1 * om1 * c12 + dom1 * s12) + m2 * (l2 * om2 * om2 + self.g * cos(phi2))

        return np.c_[x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2, T1, T2].T

    def animate(self, figsize=(8.4, 4.8), save="", show_v=False, show_a=False, wrap=False):

        phi1_full, om1_full, phi2_full, om2_full = self.full_series
        phi1, om1, phi2, om2 = self.series
        x2_full, y2_full = self.full_kinematics[3], self.full_kinematics[4]
        x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2, T1, T2 = self.kinematics
        k = self.oversample

        if wrap:
            _, series_plot, _ = self.get_cut_series(0, 2)
            phi1_plot, om1_plot, phi2_plot, om2_plot = series_plot
        else:
            phi1_plot, om1_plot, phi2_plot, om2_plot = self.full_series
        
        L = self.l1 + self.l2
        T_max = max(np.amax(np.abs(T1)), np.amax(np.abs(T2)))

        scale = figsize[0] / 14
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs
        ax.grid(ls=':')
        ax2.grid(ls=':')
        ax2.set_xlabel(r'$\varphi \; \rm [\:rad\:]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \;\rm [\:rad/s\:]$', fontsize=ftSz2)
        
        tmp, = ax.plot([-self.L*1.1, self.L*1.1], [-1.1*self.L, 1.1*self.L])
        ax.set_aspect('equal', 'datalim')
        tmp.remove()

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], 'o-', lw=2, color='C2')
        line3, = ax.plot([], [], '-', lw=1, color='grey')
        rect1 = plt.Rectangle((L * 1.050, 0), L * 0.05, T1[0] / T_max * L, color='C1', alpha=0.5)
        rect2 = plt.Rectangle((L * 1.125, 0), L * 0.05, T2[0] / T_max * L, color='C2', alpha=0.5)
        phase1, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')
        phase2, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')

        time_template = r'$t = %.1fs$'
        time_text = ax.text(0.42, 0.94, '', fontsize=ftSz2 * scale, transform=ax.transAxes)
        sector = patches.Wedge((self.L, -self.L), self.L / 15, theta1=90, theta2=90, color='lightgrey')

        kwargs = {'fontsize': ftSz3 * scale, 'wrap': True, 'transform': ax.transAxes}
        ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(self.l1), **kwargs)
        ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(self.l2), **kwargs)
        ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(self.m1), **kwargs)
        ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(self.m2), **kwargs)

        ax.text(0.61, 0.96, r'$\varphi_1  = {:.2f} $'.format(self.phi1d), **kwargs)
        ax.text(0.61, 0.92, r'$\varphi_2  = {:.2f} $'.format(self.phi2d), **kwargs)
        ax.text(0.81, 0.96, r'$\omega_1  = {:.2f} $'.format(self.om1), **kwargs)
        ax.text(0.81, 0.92, r'$\omega_2  = {:.2f} $'.format(self.om2), **kwargs)


        v_max, a_max = np.amax(v2), np.amax(ac2)
        scale_v, scale_a = 2 * self.L / (3 * v_max), 2 * self.L / (3 * a_max)
        if show_v:
            arrow_v = ax.arrow([], [], [], [], color='C3', edgecolor=None, width=self.L / 50)
        if show_a:
            arrow_a = ax.arrow([], [], [], [], color='C4', edgecolor=None, width=self.L / 50)

        ax2.plot(phi1_plot, om1_plot, color='C1', ls='-', marker='', markersize=0.1)
        ax2.plot(phi2_plot, om2_plot, color='C2', ls='-', marker='', markersize=0.1)

        #####     ================      Animation      ================      #####

        def init():
            for line in [line1, line2, line3]:
                line.set_data([], [])
            for phase in [phase1, phase2]:
                phase.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            rect1.set_bounds(L * 1.025, 0, L * 0.0, T1[0] / T_max * L)
            rect2.set_bounds(L * 1.125, 0, L * 0.0, T2[0] / T_max * L)

            res = [line1, line2, line3, phase1, time_text, phase2, sector, rect1, rect2]
            if show_v:
                arrow_v.set_data(x=x2[0], y=y2[0], dx=vx2[0] * scale_v, dy=vy2[0] * scale_v)
                res.append(arrow_v)
            if show_a:
                arrow_a.set_data(x=x2[0], y=y2[0], dx=acx2[0] * scale_a, dy=acy2[0] * scale_a)
                res.append(arrow_a)

            return tuple(res)

        def update(i):
            start = max(0, i - 250000)
            thisx, thisy = [0, x1[i]], [0, y1[i]]
            thisx2, thisy2 = [x1[i], x2[i]], [y1[i], y2[i]]

            line1.set_data(thisx, thisy)
            line2.set_data(thisx2, thisy2)
            line3.set_data([x2_full[k*start:k*i + 1]], [y2_full[k*start:k*i + 1]])
            rect1.set_bounds(L * 1.050, 0, L * 0.05, T1[i] / T_max * L)
            rect2.set_bounds(L * 1.125, 0, L * 0.05, T2[i] / T_max * L)
            phase1.set_data(phi1[i], om1[i])
            phase2.set_data(phi2[i], om2[i])

            time_text.set_text(time_template % (self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)
            ax.add_patch(rect1)
            ax.add_patch(rect2)

            res = [line1, line2, line3, phase1, time_text, phase2, sector, rect1, rect2]
            if show_v:
                arrow_v.set_data(x=x2[i], y=y2[i], dx=vx2[i] * scale_v, dy=vy2[i] * scale_v)
                res.append(arrow_v)
            if show_a:
                arrow_a.set_data(x=x2[i], y=y2[i], dx=acx2[i] * scale_a, dy=acy2[i] * scale_a)
                res.append(arrow_a)

            return tuple(res)

        def onThisClick(event):
            if event.button == 3:
                anim.event_source.stop()
                plt.close(fig)
            return

        fig.canvas.mpl_connect('button_press_event', onThisClick)
        anim = FuncAnimation(fig, update, self.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
        # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
        plt.tight_layout()

        if save == "save":
            anim.save('double_pendulum_1.html', fps=30)
        elif save == "gif":
            anim.save('./animations/trajectory.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 20.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./figures/trajectory.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()
        
        return

    def get_parameters(self):
        params1 = np.array([self.l1, self.l2, self.m1, self.m2])
        params2 = np.array([self.phi1d, self.phi2d, self.om1, self.om2])
        dcm1, dcm2 = 3, 4
        fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
        for val in params2:
            fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

        parameters = np.array([
            r"Axe x : $x_2$",
            r"Axe y : $y_2$",
            r"Axe c : $v_2$",
            "", 
            r"$T$ = {:.2f} $\rm s$".format(self.t_sim), 
            "",  # 5
            r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(self.l1, width=fmt1, dcm=dcm1),
            r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(self.l2, width=fmt1, dcm=dcm1),
            r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(self.m1, width=fmt1, dcm=dcm1),
            r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(self.m2, width=fmt1, dcm=dcm1),
            "", # 10
            r"$g$  = {:>5.2f} $\rm m/s^2$".format(self.g), 
            "",
            r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(self.phi1d, width=fmt2, dcm=dcm2),
            r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(self.phi2d, width=fmt2, dcm=dcm2),
            r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(self.om1, width=fmt2, dcm=dcm2),
            r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(self.om2, width=fmt2, dcm=dcm2),
        ])

        return parameters
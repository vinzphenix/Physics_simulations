from .simulation import *
from numpy import pi
sqrt2 = np.sqrt(2)
ftSz3 = 11

class PendulumL(Simulation):

    REQUIRED_PARAMS = ["g", "L", "h", "M", "D", "l1", "h1", "m1", "D1", "l2", "h2", "m2", "D2"]
    REQUIRED_INITIALS = ["th", "phi1", "phi2", "om", "om1", "om2"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)

        self.l = self.L - self.h
        L, l, h = self.L, self.l, self.h

        self.d, self.d1, self.d2 = sqrt2 / 2. * L * l / (L + l), (self.l1 - self.h1) / 2., (self.l2 - self.h2) / 2.
        self.IG0 = self.M / 6. * (5. * L ** 2 - 5. * L * h + h ** 2) * (L ** 2 - L * h + h ** 2) / (2. * L - h) ** 2
        self.IG1 = self.m1 / 12. * (self.l1 ** 2 + self.h1 ** 2)
        self.IG2 = self.m2 / 12. * (self.l2 ** 2 + self.h2 ** 2)

        self.thd = np.degrees(self.th)
        self.phi1d = np.degrees(self.phi1)
        self.phi2d = np.degrees(self.phi2)

        self.U0 = np.array([self.th, self.phi1, self.phi2, self.om, self.om1, self.om2])
        return

    def dynamics(self, t, u):
        # u = u.reshape(6, -1)
        th, phi1, phi2, w, w1, w2 = u
        COS1, COS2 = cos(pi / 4. + phi1 - th), cos(pi / 4. + phi2 - th)
        SIN1, SIN2 = sin(pi / 4. + phi1 - th), sin(pi / 4. + phi2 - th)

        g, M, d, l = self.g, self.M, self.d, self.l, 
        m1, m2, d1, d2 = self.m1, self.m2, self.d1, self.d2
        MAT = np.empty((3, 3, th.size))
        RHS = np.empty((3, th.size))

        MASS_A = M * d * d + l * l * (m1 + m2) + self.IG0
        MASS_D = l * m1 * d1
        MASS_E = l * m2 * d2
        MASS_B = m1 * d1 * d1 + self.IG1
        MASS_C = m2 * d2 * d2 + self.IG2

        INV_12_var = -MASS_C * MASS_D
        INV_13_var = +MASS_B * MASS_E
        INV_23_var = -MASS_D * MASS_E
        INV_11_fix = +MASS_B * MASS_C
        INV_22_fix = +MASS_A * MASS_C
        INV_22_var = -MASS_E * MASS_E
        INV_33_fix = +MASS_A * MASS_B
        INV_33_var = -MASS_D * MASS_D

        RHS_11, RHS_12, RHS_13 = -g * M * d, -g * m1 * l, -g * m2 * l
        RHS_21, RHS_31 = -g * m1 * d1, -g * m2 * d2        

        INV_12, INV_13, INV_23 = INV_12_var * COS1, INV_13_var * SIN2, INV_23_var * SIN2 * COS1
        INV_22, INV_33 = INV_22_fix + INV_22_var * SIN2 * SIN2, INV_33_fix + INV_33_var * COS1 * COS1
        MAT[0, 0], MAT[0, 1], MAT[0, 2] = INV_11_fix, INV_12, INV_13
        MAT[1, 0], MAT[1, 1], MAT[1, 2] = INV_12, INV_22, INV_23
        MAT[2, 0], MAT[2, 1], MAT[2, 2] = INV_13, INV_23, INV_33

        RHS[0] = MASS_D * SIN1 * w1 * w1 + MASS_E * COS2 * w2 * w2 + RHS_11 * sin(th) + RHS_12 * sin(
                th - pi / 4.) + RHS_13 * cos(th - pi / 4.) - self.D * w
        RHS[1] = -MASS_D * SIN1 * w * w + RHS_21 * sin(phi1) - self.D1 * w1
        RHS[2] = -MASS_E * COS2 * w * w + RHS_31 * sin(phi2) - self.D2 * w2

        delta = MASS_A * MASS_B * MASS_C \
                - MASS_B * MASS_E * MASS_E * SIN2 * SIN2 \
                - MASS_C * MASS_D * MASS_D * COS1 * COS1
        res = np.einsum("ijk,jk->ik", MAT, RHS) / delta

        # return np.c_[w, w1, w2, res[0], res[1], res[2]]
        if t < 0.:
            return np.vstack((w, w1, w2, res[0], res[1], res[2]))
        else:
            return np.hstack((w, w1, w2, res[0], res[1], res[2]))
        return
    
    def compute_kinematics(self):
        th, phi1, phi2, om, om1, om2 = self.full_series
        _, _, _, d2th, d2phi1, d2phi2 = self.dynamics(-1., np.array([th, phi1, phi2, om, om1, om2]))

        x = self.d * sin(th)
        y = - self.d * cos(th)
        vx, vy = self.d * cos(th) * om, self.d * sin(th) * om
        d2x = -self.d * sin(th) * om*om + self.d * cos(th) * d2th
        d2y = self.d * cos(th) * om*om + self.d * sin(th) * d2th

        v1x = self.l * om * cos(th - pi / 4) + self.h1 * om1 * cos(phi1)
        v1y = self.l * om * sin(th - pi / 4) + self.h1 * om1 * sin(phi1)
        v2x = - self.l * om * sin(th - pi / 4) + self.h2 * om2 * cos(phi2)
        v2y = + self.l * om * cos(th - pi / 4) + self.h2 * om2 * sin(phi2)

        return np.c_[x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y].T

    def animate(self, show_v=False, show_a=False, phaseSpace=0, save=""):
        """
        Creates and displays the animation of the system
        :param sim: object of the class Simulation with all parameters needed
        :param time_series: the vectors containing the solutions th, phi... for every time step
        :param arrow: (boolean) display or not the acceleration of the L-shaped pendulum
        :param phaseSpace: (integer) choose the display of the phase space diagram
        :param save: choose the way of saving (or not) the animation
        """
        L, l, l1, l2, h, h1, h2 = self.L, self.l, self.l1, self.l2, self.h, self.h1, self.h2
        d, d1, d2, M, m1, m2 = self.d, self.d1, self.d2, self.M, self.m1, self.m2
        
        th_full, phi1_full, phi2_full, om_full, om1_full, om2_full = self.full_series
        th, phi1, phi2, om, om1, om2 = self.series
        x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y = self.kinematics
        x_full, y_full = self.full_kinematics[:2]
        a_max = np.amax(np.hypot(d2x, d2y))
        
        k = self.oversample

        Q1x, Q1y = l * sin(th - pi / 4), - l * cos(th - pi / 4)
        Q2x, Q2y = l * cos(th - pi / 4), + l * sin(th - pi / 4)

        Ax, Ay = -h / sqrt2 * cos(th), -h / sqrt2 * sin(th)
        alpha_A = np.degrees(pi / 4. + th)
        width_A, height_A = h, -L
        Bx, By = Ax, Ay
        alpha_B = np.degrees(-pi / 4. + th)
        width_B, height_B = h, -l
        Cx, Cy = Q1x - h1 / sqrt2 * cos(phi1 - pi / 4.), Q1y - h1 / sqrt2 * sin(phi1 - pi / 4)
        alpha_C = np.degrees(phi1)
        width_C, height_C = h1, -l1
        Dx, Dy = Q2x - h2 / sqrt2 * cos(phi2 - pi / 4.), Q2y - h2 / sqrt2 * sin(phi2 - pi / 4)
        alpha_D = np.degrees(phi2)
        width_D, height_D = h2, -l2

        # Display the energy (conservation) of the system
        # E_K = 0.5 * (M * d * d + l * l * (m1 + m2) + self.IG0) * om_full * om_full + 0.5 * (
        #         m1 * d1 * d1 + self.IG1
        #     ) * om1_full * om1_full + 0.5 * (m2 * d2 * d2 + self.IG2) * om2_full * om2_full + l * om_full * (
        #         m1 * d1 * om1_full * cos(pi / 4. + phi1_full - th_full) \
        #         - m2 * d2 * om2_full * sin(pi / 4. + phi2_full - th_full)
        #     )
        # E_U = self.g * (
        #         M * d * (1 - cos(th_full)) + m1 * l * (1 - cos(th_full - pi / 4)) \
        #         + m2 * l * (sin(th_full - pi / 4) + 1.) \
        #         + m1 * d1 * (1 - cos(phi1_full)) + m2 * d2 * (1 - cos(phi2_full))
        #     )
        # plt.stackplot(self.full_t, E_K, E_U)
        # plt.show()

        #####     ================      Création de la figure      ================      #####

        fig, axs = plt.subplots(1, 2, figsize=(14., 7.))
        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

        ax = axs[0]
        ax.axis([-1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2)), -1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2))])
        ax.set_aspect("equal")
        ax.grid(ls=':')

        ax2 = axs[1]
        ax2.grid(ls=':')
        ax2.set_xlabel(r'$\varphi \; \rm [rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \; \rm [rad \: / \: s]$', fontsize=ftSz2)

        M_max = np.amax(np.array([M, m1, m2]))
        rect1 = plt.Rectangle((Ax[0], Ay[0]), width_A, height_A, alpha_A[0], color='C0', alpha=M_max/M_max, zorder=0)
        rect2 = plt.Rectangle((Bx[0], By[0]), width_B, height_B, alpha_B[0], color='C0', alpha=M_max/M_max, zorder=0)
        rect3 = plt.Rectangle((Cx[0], Cy[0]), width_C, height_C, alpha_C[0], color='C1', alpha=M_max/M_max, zorder=2)
        rect4 = plt.Rectangle((Dx[0], Dy[0]), width_D, height_D, alpha_D[0], color='C2', alpha=M_max/M_max, zorder=4)
        for rect in [rect1, rect2, rect3, rect4]:
            ax.add_patch(rect)

        point, = ax.plot([], [], marker='o', ms=5, color='black', zorder=1)
        point1, = ax.plot([], [], marker='o', ms=5, color='black', zorder=3)
        point2, = ax.plot([], [], marker='o', ms=5, color='black', zorder=5)

        phase, = ax2.plot([], [], marker='o', ms=8, color='C0', zorder=3)
        phase1, = ax2.plot([], [], marker='o', ms=8, color='C1', zorder=2)
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C2', zorder=1)

        v_max, a_max = np.amax(np.hypot(vx, vy)), np.amax(np.hypot(d2x, d2y))
        scale_v, scale_a = self.L / (2 * v_max), self.L / (2 * a_max)
        if show_v:
            arrow_v = ax.arrow([], [], [], [], color='C3', edgecolor=None, width=self.L / 50)
        if show_a:
            arrow_a = ax.arrow([], [], [], [], color='C4', edgecolor=None, width=self.L / 50)

        time_template = r'$t \;\:\: = {:.2f} \; s$' if save == "snapshot" else r'$t \;\:\: = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.45, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
        sector = patches.Wedge((0.5, 0.85), 0.04, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)

        kw = dict(fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.17, 0.96, r'$L  = {:.2f} \,m$'.format(L), **kw)
        ax.text(0.17, 0.93, r'$h  = {:.2f} \,m$'.format(h), **kw)
        ax.text(0.17, 0.90, r'$m  = {:.2f} \,kg$'.format(M), **kw)

        ax.text(0.02, 0.96, r'$l_1  = {:.2f} \,m$'.format(l1), **kw)
        ax.text(0.02, 0.93, r'$h_1  = {:.2f} \,m$'.format(h1), **kw)
        ax.text(0.02, 0.90, r'$m_1  = {:.2f} \,kg$'.format(m1), **kw)

        ax.text(0.02, 0.85, r'$l_2  = {:.2f} \,m$'.format(l2), **kw)
        ax.text(0.02, 0.82, r'$h_2  = {:.2f} \,m$'.format(h2), **kw)
        ax.text(0.02, 0.79, r'$m_2  = {:.2f} \,kg$'.format(m2), **kw)

        ax.text(0.70, 0.96, r'$\vartheta  = {:.2f} $'.format(self.th), **kw)
        ax.text(0.70, 0.93, r'$\varphi_1  = {:.2f} $'.format(self.phi1), **kw)
        ax.text(0.70, 0.90, r'$\varphi_2  = {:.2f} $'.format(self.phi2), **kw)
        ax.text(0.85, 0.96, r'$\omega  \;\,  = {:.2f} $'.format(self.om), **kw)
        ax.text(0.85, 0.93, r'$\omega_1  = {:.2f} $'.format(self.om1), **kw)
        ax.text(0.85, 0.90, r'$\omega_2  = {:.2f} $'.format(self.om2), **kw)
        if phaseSpace == 0:
            ax2.plot(th_full, om_full, color='C0', label='pendule L', zorder=3)
            ax2.plot(phi1_full, om1_full, ls='-', marker='', ms=1, color='C1', label='masse 1', zorder=2)
            ax2.plot(phi2_full, om2_full, ls='-', marker='', ms=1, color='C2', label='masse 2', zorder=1)
            ax2.legend(fontsize=0.5 * (ftSz2 + ftSz3))
        else:
            ax2.plot(phi1, phi2, color='C1')
            ax2.plot(om1, om2, color='C2')

        #####     ================      Animation      ================      #####
            
        def init():
            point.set_data([], [])
            point1.set_data([], [])
            point2.set_data([], [])
            phase1.set_data([], [])
            phase2.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)

            res = [rect1, rect2, rect3, rect4, point, point1, point2, phase1, phase2, time_text, sector]

            if phaseSpace == 0:
                phase.set_data([], [])
                res.append(phase)
            if show_v:
                arrow_v.set_data(x=x[0], y=y[0], dx=vx[0] * scale_v, dy=vy[0] * scale_v)
                res.append(arrow_v)
            if show_a:
                arrow_a.set_data(x=x[0], y=y[0], dx=d2x[0] * scale_a, dy=d2y[0] * scale_a)
                res.append(arrow_a)
            
            return tuple(res)

        def update(i):
            res = []

            rect1.set(xy=(Ax[i], Ay[i]), angle=alpha_A[i])
            rect2.set(xy=(Bx[i], By[i]), angle=alpha_B[i])
            rect3.set(xy=(Cx[i], Cy[i]), angle=alpha_C[i])
            rect4.set(xy=(Dx[i], Dy[i]), angle=alpha_D[i])
            res += [rect1, rect2, rect3, rect4]

            point.set_data(0, 0)
            point1.set_data(Q1x[i], Q1y[i])
            point2.set_data(Q2x[i], Q2y[i])
            res += [point, point1, point2]

            if phaseSpace == 0:
                phase.set_data(th[i], om[i])
                phase1.set_data(phi1[i], om1[i])
                phase2.set_data(phi2[i], om2[i])
                res += [phase, phase1, phase2]
            else:
                phase1.set_data(phi1[i], phi2[i])
                phase2.set_data(om1[i], om2[i])
                res += [phase1, phase2]

            if show_v:
                arrow_v.set_data(x=x[i], y=y[i], dx=vx[i] * scale_v, dy=vy[i] * scale_v)
                res.append(arrow_v)
            if show_a:
                arrow_a.set_data(x=x[i], y=y[i], dx=d2x[i] * scale_a, dy=d2y[i] * scale_a)
                res.append(arrow_a)

            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)
            res += [time_text, sector]

            return tuple(res)

        # self.n_frames //= 10 if save == "gif" else 1
        anim = FuncAnimation(
            fig, update, self.n_frames, interval=20, 
            blit=True, init_func=init, repeat_delay=3000
        )
        fig.tight_layout()

        if save == "save":
            anim.save('Pendule_L_1', fps=30)
        elif save == "gif":
            anim.save('./pendulum_L.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 12.5
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./pendulum_L.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

# def solve_equations_of_motion(sim):
#     """
#     solve the equations of motion of this system
#     :param sim: object of the class Simulation with all parameters needed
#     :return: the vectors containing the solutions th, phi... for every time step
#     """
        # if _ < 0.:
        #     return np.vstack((w, w1, w2, res[0], res[1], res[2]))
        # else:
        #     return np.hstack((w, w1, w2, res[0], res[1], res[2]))

    # sol = odeint(f, U0, t, tfirst=True, atol=1.e-9, rtol=1.e-9, args=(MATRIX, VECTOR))
    # sol = solve_ivp(f, [0, sim.t_sim], U0, method="LSODA", t_eval=t, atol=1.e-12, rtol=1.e-12, args=(MATRIX, VECTOR))

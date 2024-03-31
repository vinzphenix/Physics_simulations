from .simulation import *

class TriplePendulum(Simulation):

    REQUIRED_PARAMS = ["g", "l1", "l2", "l3", "m1", "m2", "m3"]
    REQUIRED_INITIALS = ["phi1", "om1", "phi2", "om2", "phi3", "om3"]

    def __init__(self, setup, params, initials):
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        self.L = self.l1 + self.l2 + self.l3
        self.M = self.m1 + self.m2 + self.m3
        self.m = self.m2 + self.m3
        
        self.phi1d = np.degrees(self.phi1)
        self.phi2d = np.degrees(self.phi2)
        self.phi3d = np.degrees(self.phi3)

        self.U0 = np.array([self.phi1, self.om1, self.phi2, self.om2, self.phi3, self.om3])
        return
    
    def dynamics(self, t, u):

        th1, w1, th2, w2, th3, w3 = u
        l1, l2, l3 = self.l1, self.l2, self.l3
        m1, m2, m3 = self.m1, self.m2, self.m3
        M, m = self.M, self.m
        g = self.g

        C31, S31 = cos(th3 - th1), sin(th3 - th1)
        C32, S32 = cos(th3 - th2), sin(th3 - th2)
        C21, S21 = cos(th2 - th1), sin(th2 - th1)

        Num1 = (m3 * C31 * C32 - m * C21) / (m - m3 * C32 * C32) * (l1 * w1 * w1 * (
                -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
            th2) + m3 * g * sin(th3) * C32)
        Num2 = m * l2 * w2 * w2 * S21 + m3 * C31 * (
            g * sin(th3) + l2 * w2 * w2 * S32 + l1 * w1 * w1 * S31) + m3 * l3 * w3 * w3 * S31 - M * g * sin(th1)
        Den = l1 * (M - (m * C21 - m3 * C31 * C32) * (m * C21 - m3 * C31 * C32) / (m - m3 * C32 * C32) - m3 * C31 * C31)

        f1 = (Num1 + Num2) / Den
        f2 = (f1 * (-l1 * m * C21 + m3 * l1 * C31 * C32) + l1 * w1 * w1 * (
                -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
            th2) + m3 * g * sin(th3) * C32) / (l2 * (m - m3 * C32 * C32))
        f3 = 1. / l3 * (-g * sin(th3) - l2 * w2 * w2 * S32 - l2 * f2 * C32 - l1 * w1 * w1 * S31 - l1 * f1 * C31)

        return np.array([w1, f1, w2, f2, w3, f3])

    def compute_kinematics(self):

        l1, l2, l3 = self.l1, self.l2, self.l3
        phi1, om1, phi2, om2, phi3, om3 = self.full_series

        x1, y1 = 0. + l1 * sin(phi1), 0. - l1 * cos(phi1)
        x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
        x3, y3 = x2 + l3 * sin(phi3), y2 - l3 * cos(phi3)

        vx2 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2)
        vy2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
        v2 = np.hypot(vx2, vy2)

        vx3 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2) - l3 * om3 * sin(phi3)
        vy3 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2) + l3 * om3 * cos(phi3)
        v3 = np.hypot(vx3, vy3)
        
        return np.c_[x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3].T


    def animate(self, figsize=(13., 6.), save=""):
        
        phi1_full, om1_full, phi2_full, om2_full, phi3_full, om3_full = self.full_series
        x2_full, y2_full, x3_full, y3_full = self.full_kinematics[2:6]
        phi1, om1, phi2, om2, phi3, om3 = self.series
        x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3 = self.kinematics
        k = self.oversample

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axs[0], axs[1]

        xmin, xmax = min(np.amin(x1), np.amin(x2), np.amin(x3)), max(np.amax(x1), np.amax(x2), np.amax(x3))
        ymin, ymax = -self.L, max(np.amax(y1), np.amax(y2), np.amax(y3), 0.)
        xmax = max(-xmin, xmax)
        xmax = max(xmax, 0.5 * (ymax-ymin))
        ax.axis([-1.15 * xmax, 1.15 * xmax, ymin - 0.1 * xmax, ymin + 2.2 * xmax])
        # ax.axis([-1.1*L, 1.1*L, -1.1*L, 1.1*L])
        ax.set_aspect("equal")
        ax2.set_xlabel(r'$\varphi \rm \;[rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \rm \;[rad/s]$', fontsize=ftSz2)
        ax.grid(ls=':')
        ax2.grid(ls=':')

        line1, = ax.plot([], [], 'o-', lw=2, color='C1')
        line2, = ax.plot([], [], 'o-', lw=2, color='C2')
        line3, = ax.plot([], [], 'o-', lw=2, color='C3')
        line4, = ax.plot([], [], '-', lw=1, color='grey')
        line5, = ax.plot([], [], '-', lw=1, color='lightgrey')

        sector = patches.Wedge((xmax, ymin + 0.1 / 2. * xmax), xmax / 10., theta1=90, theta2=90, color='lightgrey')

        phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')
        phase3, = ax2.plot([], [], marker='o', ms=8, color='C0')

        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.50, 0.95, '', fontsize=ftSz2, transform=ax.transAxes, ha="center")

        ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(self.l1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(self.l2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.02, 0.88, r'$l_3  = {:.2f} \: \rm m$'.format(self.l3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(self.m1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(self.m2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.18, 0.88, r'$m_3  = {:.2f} \: \rm kg$'.format(self.m3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

        ax.text(0.64, 0.96, r'$\varphi_1  = {:.2f}$'.format(self.phi1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.64, 0.92, r'$\varphi_2  = {:.2f}$'.format(self.phi2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.64, 0.88, r'$\varphi_3  = {:.2f}$'.format(self.phi3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.84, 0.96, r'$\omega_1  = {:.2f}$'.format(self.om1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.84, 0.92, r'$\omega_2  = {:.2f}$'.format(self.om2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
        ax.text(0.84, 0.88, r'$\omega_3  = {:.2f}$'.format(self.om3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

        ax2.plot(phi1_full, om1_full, color='C1')
        ax2.plot(phi2_full, om2_full, color='C2')
        ax2.plot(phi3_full, om3_full, color='C3')

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            line4.set_data([], [])
            line5.set_data([], [])
            sector.set_theta1(90)
            phase1.set_data([], [])
            phase2.set_data([], [])
            phase3.set_data([], [])
            time_text.set_text('')
            return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

        def update(i):
            start = max(0, i - 10000)
            thisx, thisx2, thisx3 = [0, x1[i]], [x1[i], x2[i]], [x2[i], x3[i]]
            thisy, thisy2, thisy3 = [0, y1[i]], [y1[i], y2[i]], [y2[i], y3[i]]

            line1.set_data(thisx, thisy)
            line2.set_data(thisx2, thisy2)
            line3.set_data(thisx3, thisy3)
            line4.set_data([x3_full[k*start:k*i + 1]], [y3_full[k*start:k*i + 1]])
            line5.set_data([x2_full[k*start:k*i + 1]], [y2_full[k*start:k*i + 1]])

            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)
            time_text.set_text(time_template.format(self.t[i]))

            phase1.set_data(phi1[i], om1[i])
            phase2.set_data(phi2[i], om2[i])
            phase3.set_data(phi3[i], om3[i])

            return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

        anim = FuncAnimation(fig, update, self.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
        fig.tight_layout()

        if save == "save":
            anim.save('triple_pendulum_2.html', fps=30)
        elif save == "gif":
            anim.save('./triple_pendulum.gif', writer=PillowWriter(fps=20))
        elif save == "snapshot":
            t_wanted = 20.
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./triple_pendulum.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()
        
        return

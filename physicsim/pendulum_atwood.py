from .simulation import *

class AtwoodPendulum(Simulation):

    REQUIRED_PARAMS = ["g", "M", "m"]
    REQUIRED_INITIALS = ["r", "dr", "th", "om"]

    def __init__(self, setup, params, initials):
                
        # sets the attributes from setup, params, initials
        super().__init__(setup, params, initials, self.REQUIRED_PARAMS, self.REQUIRED_INITIALS)
        
        self.mu = self.M / self.m
        self.thd = np.degrees(self.th)

        # initial conditions
        self.U0 = np.array([self.r, self.dr, self.th, self.om])
        
        return
    
    def dynamics(self, t, u):
        m, g, M = self.m, self.g, self.M
        r_, dr_, th_, om_ = u
        f1 = (m * r_ * om_ * om_ + g * (-M + m * cos(th_))) / (M + m)
        f2 = -1 / r_ * (2 * dr_ * om_ + g * sin(th_))
        return np.array([dr_, f1, om_, f2])
    
    def compute_kinematics(self):
        g, M, m = self.g, self.M, self.m
        r, dr, th, om = self.full_series

        d = np.mean(r)
        L = max(r) + d * 1.2

        x1, y1 = -d * np.ones_like(r), r + d - L
        x2, y2 = r * sin(th), -r * cos(th)

        vx = dr * sin(th) + r * om * cos(th)
        vy = -dr * cos(th) + r * om * sin(th)
        v = np.hypot(vx, vy)

        _, ddr, _, dom = self.dynamics(self.t, [r, dr, th, om])
        # ddr = (m * r * om * om + g * (-M + m * cos(th))) / (M + m)
        # dom = -1 / r * (2 * dr * om + g * sin(th))
        acx = ddr * sin(th) + dr * cos(th) * om - r * om * om * sin(th) + (r * dom + dr * om) * cos(th)
        acy = -ddr * cos(th) + dr * sin(th) * om + r * om * om * cos(th) + (r * dom + dr * om) * sin(th)
        a = np.hypot(acx, acy)

        return np.c_[x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a].T


    def animate(self, figsize=(13., 6.), save=""):
        r_full, dr_full, th_full, om_full = self.full_series
        r, dr, th, om = self.series
        x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = self.kinematics
        x2_full, y2_full = self.full_kinematics[2], self.full_kinematics[3]

        L_X, L_Y = np.amax(x2) - np.amin(x2), np.amax(y2) - np.amin(y2)
        x_m, y_m = np.amin(x2) - 0.2 * L_X, np.amin(y2) - 0.2 * L_Y
        x_M, y_M = np.amax(x2) + 0.2 * L_X, np.amax(y2) + 0.2 * L_Y

        d = np.mean(r)
        x_m = min(-d * 1.1, x_m)
        if abs(np.amin(y2)) < L_X:
            y_m = x_m
            y_M = x_M

        plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
        fig, axs = plt.subplots(1, 2, figsize=(14., 7.))  # , constrained_layout=True)
        ax, ax2 = axs[0], axs[1]

        tmp, = ax.plot([x_m, x_M], [y_m, y_M])
        ax.set_aspect("equal", "datalim")
        tmp.remove()
        ax.grid(ls=':')
        ax2.grid(ls=':')
        ax2.set_xlabel(r'$\vartheta \;\; \rm [rad]$', fontsize=ftSz2)
        ax2.set_ylabel(r'$\omega \;\; \rm [rad\,/\,s]$', fontsize=ftSz2)

        line,  = ax.plot([], [], 'o-', lw=2.5, color='C1', zorder=10)
        line2, = ax.plot([], [], '-', lw=1, color='grey')
        phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
        phase2, = ax2.plot([], [], marker='o', ms=8, color='C1', alpha=0.)

        time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
        time_text = ax.text(0.40, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
        sector = patches.Wedge((x_M - L_X/10, x_m + L_X/10), L_X/20, theta1=90, theta2=90, color='lightgrey')

        ax.text(0.04, 0.94, r'$\mu  = {:.2f}$'.format(self.mu), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

        ftSz4 = ftSz3 * 0.9
        ax.text(0.66, 0.96, r'$r  = {:.2f} $'.format(self.r), fontsize=ftSz4, transform=ax.transAxes)
        ax.text(0.66, 0.92, r'$\dot{{r}}  = {:.2f} $'.format(self.dr), fontsize=ftSz4, transform=ax.transAxes)
        ax.text(0.80, 0.96, r'$\vartheta  = {:.2f} \:\rm deg$'.format(self.thd), fontsize=ftSz4, transform=ax.transAxes)
        ax.text(0.80, 0.92, r'$\omega  = {:.2f} \:\rm rad/s$'.format(self.om), fontsize=ftSz4, transform=ax.transAxes)

        ax2.plot(th_full, om_full, color='C0')
        # ax2.plot(r_full, dr_full, color='C1')

        #####     ================      Animation      ================      #####

        def init():
            line.set_data([], [])
            line2.set_data([], [])
            phase1.set_data([], [])
            phase2.set_data([], [])
            time_text.set_text('')
            sector.set_theta1(90)
            return line, line2, phase1, phase2, time_text, sector

        def update(i):
            start = max(0, i - int(self.slowdown * self.fps * 50.))  # display history of last ... seconds
            thisx = [x2[i], 0, x1[i], x1[i]]
            thisy = [y2[i], 0, 0, y1[i]]

            line.set_data(thisx, thisy)
            line2.set_data([x2_full[k * start:k * i + 1]], [y2_full[k * start: k * i + 1]])
            phase1.set_data(th[i], om[i])
            phase2.set_data(r[i], dr[i])

            time_text.set_text(time_template.format(self.t[i]))
            sector.set_theta1(90 - 360 * self.t[i] / self.t_sim)
            ax.add_patch(sector)

            return line, line2, phase1, phase2, time_text, sector

        fig.tight_layout()
        k = self.oversample
        # self.n_frames //= 20 if save == "gif" else 1
        anim = FuncAnimation(fig, update, self.n_frames + 1, interval=5., blit=True, init_func=init, repeat_delay=5000)
        # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)

        if save == "html":
            anim.save('atwood.html', writer=HTMLWriter(fps=self.fps))
        elif save == "gif":
            # noinspection PyTypeChecker
            anim.save('./atwood.gif', writer=PillowWriter(fps=self.fps))
        elif save == "mp4":
            anim.save(f"./atwood.mp4", writer=FFMpegWriter(fps=self.fps))
        elif save == "snapshot":
            t_wanted, self.oversample = 20., 1
            t_idx = np.argmin(np.abs(self.t - t_wanted))
            update(t_idx)
            fig.savefig("./atwood.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()

    def get_parameters(self):
        params = np.array([self.r, self.dr, self.thd, self.om])
        dcm1, dcm2 = 5, 3
        fmt2 = 1 + 1 + dcm2
        for val in params:
            fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

        parameters = np.array([
            r"Axe x : $x_2$",
            r"Axe y : $y_2$",
            r"Axe c : $v_2$", 
            "",
            r"$\Delta t$ = {:.2f} $\rm s$".format(self.t_sim), 
            "",  # 5
            r"$\mu$ = {:.{dcm}f}".format(self.M / self.m, dcm=dcm1),
            "", 
            r"$g$ = {:.2f} $\rm m/s^2$".format(self.g), 
            "",  # 9
            r"$r \;\,\,$ = {:>{width}.{dcm}f} $\rm m$".format(self.r, width=fmt2, dcm=dcm2),
            r"$dr$ = {:>{width}.{dcm}f} $\rm m/s$".format(self.dr, width=fmt2, dcm=dcm2),
            r"$\vartheta \;\,$ = {:>{width}.{dcm}f} $\rm deg$".format(self.thd, width=fmt2, dcm=dcm2),
            r"$\omega \;\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(self.om, width=fmt2, dcm=dcm2)
        ])
        
        return parameters
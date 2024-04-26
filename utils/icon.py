import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from numpy import sin, cos, tan, pi
from operator import sub

from physicsim.pendulum_atwood import AtwoodPendulum
from physicsim.cylinder_slide import Cylinder
from physicsim.pendulum_driven import DrivenPendulum
from physicsim.pendulum_2 import DoublePendulum
from physicsim.pendulum_3 import TriplePendulum
from physicsim.pendulum_elastic import PendulumElastic
from physicsim.pendulum_vertical import VerticalPendulum
from physicsim.pendulum_horiztontal import HorizontalPendulum

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def get_straight_arrow(org, vec, strength):
    vec /= np.hypot(*vec)    
    arrow_pts = np.c_[org + 0.1*strength*vec, org + 0.75*strength*vec].T
    
    head = org + 1.0*strength*vec
    head_base = org + 0.75*strength*vec
    vec2 = head - head_base
    vec2 = np.array([-vec2[1], vec2[0]])
    head_l = head_base + 0.5 * vec2
    head_r = head_base - 0.5 * vec2
    arrow_head = np.c_[head_l, head, head_r].T
    
    return arrow_pts, arrow_head


def get_double_straight_arrow(org, dst):
    arrow_pts = np.c_[org * 0.75 + 0.25 * (dst), org * 0.25 + 0.75*dst].T
    arrow_heads = []
    for end, vec in zip([dst, org], [dst-org, org-dst]):
        head = end
        head_base = end - 0.25*vec
        vec2 = head - head_base
        vec2 = np.array([-vec2[1], vec2[0]])
        head_l = head_base + 0.5 * vec2
        head_r = head_base - 0.5 * vec2
        arrow_heads.append(np.c_[head_l, head, head_r].T)
    return arrow_pts, *arrow_heads
 

def get_curved_arrow(org, dst, radius, strength, head_pct=0.30):
    """
    draws a circle arrow going through the midpoint of (org, dst)
    the circle center is at distance radius from the midpoint towards org
    the strength indicates the length of the arrow
    """
    angle_a = -0.4*strength
    angle_b = 0.6*strength
    
    mid = (dst + org) * 0.5
    vec = (dst - org) / np.hypot(*(dst - org))
    
    n_pts = 50
    alpha = np.linspace(angle_a, angle_b, n_pts)
    rot_mat = np.empty((n_pts, 2, 2))
    
    #arrow_dst = mid + arrow_r * (-vec + rot_mat_dst @ vec)
    rot_mat[:, 0, 0] = +np.cos(alpha) - 1.
    rot_mat[:, 0, 1] = -np.sin(alpha)
    rot_mat[:, 1, 0] = -rot_mat[:, 0, 1]
    rot_mat[:, 1, 1] = +rot_mat[:, 0, 0]
    arrow_pts = mid + radius * np.einsum('kij,j->ki', rot_mat, vec)
    head = arrow_pts[n_pts-1]
    idx_base = int(n_pts * (1 - head_pct))
    head_base = arrow_pts[idx_base]
    arrow_pts = arrow_pts[:idx_base+1]
    
    vec2 = head - head_base
    vec2 = np.array([-vec2[1], vec2[0]])
    head_l = head_base + 0.5 * vec2
    head_r = head_base - 0.5 * vec2
    arrow_head = np.c_[head_l, head, head_r].T
    return arrow_pts, arrow_head


def get_display(ax):
    figW, figH = ax.get_figure().get_size_inches()  # total figure size
    _, _, w, h = ax.get_position().bounds  # axis size on figure
    return figW * w, figH * h


def compute_scaling(ax, icon_bounds, fx, fy, pad):
    """
    Determines if the scaling is tight at the width percentage constraint,
    or at the height percentage constraint. Returns the adequate scaling.
    """
    figW, figH = ax.get_figure().get_size_inches()  # total figure size
    _, _, w, h = ax.get_position().bounds  # axis size on figure
    disp_ratio = (figH * h) / (figW * w)  # ratio of display units
    
    icon_xmin, icon_xmax, icon_ymin, icon_ymax = icon_bounds
    
    scale_x = (fx-pad) / (icon_xmax - icon_xmin)  # x-scaling dictates the tsfm
    scale_y = (fy-pad) / (icon_ymax - icon_ymin)  # y-scaling dictates the tsfm
    
    # choose the scaling that respects both the width and height percentage
    if scale_x < scale_y * disp_ratio:
        scale_y = scale_x / disp_ratio
        pad_x, pad_y = pad, pad / disp_ratio
    else:
        scale_x = scale_y * disp_ratio
        pad_x, pad_y = pad * disp_ratio, pad
    
    return scale_x, scale_y, pad_x, pad_y


def map_xy(x, y, ax, shift_x, shift_y, scale_x, scale_y, pad_x, pad_y):
    """
    Maps the icon coordinates to the plot coordinates
    """
    # get axis limits
    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()
    # map icon data to pre-existing axis data, with padding
    xnew = ax_xmax + (ax_xmax - ax_xmin) * ((x - shift_x) * scale_x - pad_x)
    ynew = ax_ymin + (ax_ymax - ax_ymin) * ((y - shift_y) * scale_y + pad_y)
    return xnew, ynew


def get_color_palette(c1):
    # c1 : moving points
    c2 = tuple(np.r_[0.85*np.array(c1)[:3], c1[3]])  # segments
    c3 = tuple(np.r_[0.65*np.array(c1)[:3], c1[3]])  # fixed points
    c4 = tuple(np.r_[0.85*np.array(c1)[:3], 0.85*c1[3]])  # arrows
    return c2, c3, c4
    
def icon_dbl_pendulum(ax, sim, fx, fy, pad, c1, lw, driven=False):
    l1, l2 = sim.l1, sim.l2
    if not driven:
        m1, m2 = sim.m1, sim.m2
    else:
        m1, m2 = 1., 1.
    phi1, phi2, om1, om2 = sim.phi1, sim.phi2, sim.om1, sim.om2
    if hasattr(sim, 'l3'):
        l3, m3 = sim.l3, sim.m3
        phi3, om3 = sim.phi3, sim.om3
    else:
        l3, m3 = -1, 0.
        phi3, om3 = 0., 0.
    w_disp, h_disp = get_display(ax)
    
    x0, y0 = 0., 0.
    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
    x3, y3 = (x2 + l3 * sin(phi3), y2 - l3 * cos(phi3)) if l3 > 0. else (0., 0.)
    orgs = [np.array([0., 0.]), np.array([x1, y1]), np.array([x2, y2])]
    dsts = [np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])]
    oms = [om1, om2, om3]
    
    data_xmin, data_xmax = min(0., x1, x2, x3), max(0., x1, x2, x3)
    data_ymin, data_ymax = min(0., y1, y2, y3), max(0., y1, y2, y3)
    delta_x, delta_y = data_xmax - data_xmin, data_ymax - data_ymin
    if delta_x < 1e-6:
        data_xmin, data_xmax = data_xmin - delta_y*0.1, data_xmax + delta_y*0.1
    if delta_y < 1e-6:
        data_ymin, data_ymax = data_ymin - delta_x*0.1, data_ymax + delta_x*0.1
    icon_bounds = [data_xmin, data_xmax, data_ymin, data_ymax]
    
    scl_x, scl_y, pad_x, pad_y = compute_scaling(ax, icon_bounds, fx, fy, pad)

    args = [ax, data_xmax, data_ymin, scl_x, scl_y, pad_x, pad_y]
    ox, oy = map_xy(0., 0., *args)
    x1, y1 = map_xy(x1, y1, *args)
    x2, y2 = map_xy(x2, y2, *args)
    x3, y3 = map_xy(x3, y3, *args)
    
    c2, c3, c4 = get_color_palette(c1)
    ls = ':' if driven else '-'
    ms = 15*(0.1*max(w_disp, h_disp))
    LW = lw * (0.1*max(w_disp, h_disp))
    ax.plot(
        [ox], [oy], 's', ms=ms*np.sqrt(0.5),
        markerfacecolor=c3, markeredgecolor="none", zorder=2
    ) # center
    ax.plot([x1], [y1], 'o', ms=ms*np.sqrt(m1/(m1+m2+m3)), color=c1, zorder=1) # node 1
    ax.plot([x2], [y2], 'o', ms=ms*np.sqrt(m2/(m1+m2+m3)), color=c1, zorder=1) # node 2
    ax.plot([ox, x1], [oy, y1], ls=ls, lw=LW, color=c2, zorder=0) # segment 1
    ax.plot([x1, x2], [y1, y2], '-', lw=LW, color=c2, zorder=0) # segment 2
    if l3 > 0.:
        ax.plot([x3], [y3], 'o', ms=ms*np.sqrt(m3/(m1+m2+m3)), color=c1, zorder=1)
        ax.plot([x2, x3], [y2, y3], '-', lw=LW, color=c2, zorder=0)
    
    for i, (org, dst, om) in enumerate(zip(orgs, dsts, oms)):
        if (i == 2) and l3 < 0.:
            break
        strength = 0.15*om
        arrow_r = l1+l2
        arrow_pts, arrow_head = get_curved_arrow(org, dst, arrow_r, strength)
        a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
        arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
        ax.plot(arrow_pts_x, arrow_pts_y, ls='-', color=c4)
        ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
        
    return


def icon_atwood(ax, sim, fx, fy, pad, c1, lw):
    M, m, d, L = sim.M, sim.m, sim.d, sim.L
    r0, v0, th0, om0 = sim.r, sim.dr, sim.th, sim.om
    w_disp, h_disp = get_display(ax)
    
    x0, y0 = -d, -L+d+r0
    x1, y1 = -d, 0.
    x2, y2 = 0., 0.
    x3, y3 = r0 * sin(th0), -r0 * cos(th0)
    orgs = [np.array([x2, y2]), np.array([x3, y3])]
    dsts = [np.array([x3, y3]), np.array([x3, y3]) - np.array([x2, y2])]
    oms = [om0, v0]
    
    data_xmin, data_xmax = min(x0, x1, x2, x3), max(x0, x1, x2, x3)
    data_ymin, data_ymax = min(y0, y1, y2, y3), max(y0, y1, y2, y3)
    icon_bounds = [data_xmin, data_xmax, data_ymin, data_ymax]
    scl_x, scl_y, pad_x, pad_y = compute_scaling(ax, icon_bounds, fx, fy, pad)

    args = [ax, data_xmax, data_ymin, scl_x, scl_y, pad_x, pad_y]
    x0, y0 = map_xy(x0, y0, *args)
    x1, y1 = map_xy(x1, y1, *args)
    x2, y2 = map_xy(x2, y2, *args)
    x3, y3 = map_xy(x3, y3, *args)
    
    c2, c3, c4 = get_color_palette(c1)
    ms = 12*(0.1*max(w_disp, h_disp))
    LW = lw * (0.1*max(w_disp, h_disp))
    ax.plot(
        [x0], [y0], 'o', ms=ms*np.sqrt(M/(m+M)), zorder=1, 
        markerfacecolor=c1, markeredgecolor="none"
    ) # big mass
    ax.plot(
        [x1], [y1], 's', ms=ms*np.sqrt(0.5), zorder=1, 
        markerfacecolor=c3, markeredgecolor="none"
    ) # node
    ax.plot(
        [x2], [y2], 's', ms=ms*np.sqrt(0.5), zorder=1, 
        markerfacecolor=c3,  markeredgecolor="none"
    ) # node
    ax.plot(
        [x3], [y3], 'o', ms=ms*np.sqrt(m/(m+M)), zorder=1, 
        markerfacecolor=c1, markeredgecolor="none"
    ) # small mass
    ax.plot(
        [x0, x1, x2, x3], [y0, y1, y2, y3], '-', lw=LW, 
        color=c2, zorder=0
    ) # segment 1
    
    org, dst, om = orgs[0], dsts[0], oms[0]
    strength = 0.15*om
    arrow_r = 5.0
    arrow_pts, arrow_head = get_curved_arrow(org, dst, arrow_r, strength)
    a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
    arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
    ax.plot(arrow_pts_x, arrow_pts_y, color=c4)
    ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
    
    org, dst, om = orgs[1], dsts[1], oms[1]
    arrow_pts, arrow_head = get_straight_arrow(org, dst, v0)
    a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
    arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
    ax.plot(arrow_pts_x, arrow_pts_y, color=c4)
    ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
    return
    

def icon_mv_pendulum(ax, sim, fx, fy, pad, c1, lw, vertical=True):
    l, m, M, F, w = sim.l, sim.mp, sim.mb, sim.F, sim.w
    phi0, om0, x0, v0 = sim.phi, sim.om, sim.x, sim.dx
    w_disp, h_disp = get_display(ax)
    
    x0, y0 = 0., 0.
    x1, y1 = -l * sin(phi0), l * cos(phi0)
    x2, y2 = -l/2, -l/10
    x3, y3 = l/2, -l/10
    x4, y4 = l/2, +l/10
    x5, y5 = -l/2, +l/10
    orgs = [np.array([x0, y0]), np.array([0., 0.])]
    dsts = [np.array([x1, y1]), np.array([0., -1])]
    if vertical:
        orgs.append(np.array([-l*0.6, -l/4]))
        dsts.append(np.array([-l*0.6, +l/4]))
    else:
        orgs.append(np.array([-l/4, -l/5]))
        dsts.append(np.array([+l/4, -l/5]))
    oms = [om0, v0]
    
    data_xmin, data_xmax = min(x0, x1, x2, x3, x4, x5), max(x0, x1, x2, x3, x4, x5)
    data_ymin, data_ymax = min(y0, y1, y2, y3, y4, y5, -v0), max(y0, y1, y2, y3, y4, y5, -v0)
    icon_bounds = [data_xmin, data_xmax, data_ymin, data_ymax]
    scl_x, scl_y, pad_x, pad_y = compute_scaling(ax, icon_bounds, fx, fy, pad)

    args = [ax, data_xmax, data_ymin, scl_x, scl_y, pad_x, pad_y]
    ox, oy = map_xy(0., 0., *args)
    x1, y1 = map_xy(x1, y1, *args)
    x2, y2 = map_xy(x2, y2, *args)
    x3, y3 = map_xy(x3, y3, *args)
    x4, y4 = map_xy(x4, y4, *args)
    x5, y5 = map_xy(x5, y5, *args)

    c2, c3, c4 = get_color_palette(c1)
    ms = 8*(0.1*max(w_disp, h_disp))
    LW = lw * (0.1*max(w_disp, h_disp))
    ax.plot(
        [ox], [oy], 'o', ms=ms*np.sqrt(0.5), color=c1,
        zorder=1, markerfacecolor=c3, markeredgecolor="none"
    ) # center
    ax.plot([x1], [y1], 'o', ms=2*ms*np.sqrt(m/(m+M)), color=c1, zorder=1) # node 1
    ax.plot([ox, x1], [oy, y1], ls='-', lw=LW, color=c2, zorder=0) # segment 1
    ax.fill([x2, x3, x4, x5], [y2, y3, y4, y5], color=c3, edgecolor='none')

    org, dst, om = orgs[0], dsts[0], oms[0]
    strength = 0.10*om
    arrow_r = l
    arrow_pts, arrow_head = get_curved_arrow(org, dst, arrow_r, strength)
    a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
    arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
    ax.plot(arrow_pts_x, arrow_pts_y, ls='-', color=c4)
    ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
    
    org, dst, om = orgs[1], dsts[1], oms[1]
    arrow_pts, arrow_head = get_straight_arrow(org, dst, v0)
    a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
    arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
    ax.plot(arrow_pts_x, arrow_pts_y, color=c4)
    ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')

    org, dst = orgs[2], dsts[2]
    arrow_pts, arrow_head, arrow_tail = get_double_straight_arrow(org, dst)
    # ws(arrow_pts, arrow_head, arrow_tail)
    arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
    ax.plot(arrow_pts_x, arrow_pts_y, color=c4)
    for arrow in [arrow_head, arrow_tail]:
        a_head_x, a_head_y = map_xy(arrow[:, 0], arrow[:, 1], *args)
        ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
    
    return


def icon_cylinder(ax, sim, fx, fy, pad, c1, lw):
    C, a, m, M, R, L = sim.C, sim.alpha, sim.m, sim.M, sim.R, sim.L
    th0, om0, x0, v0 = sim.th, sim.om, sim.x, sim.dx
    w_disp, h_disp = get_display(ax)
    
    x0, y0 = 0., 0.  # disk center
    x1, y1 = L * cos(th0 + a), - L * sin(th0 + a)  # pendulum node

    slope_size = 5*R
    x2, y2 = -slope_size/2., 0. - R / cos(a) - tan(a) * (-slope_size/2. - 0.)  # slope top
    x3, y3 = +slope_size/2., 0. - R / cos(a) - tan(a) * (+slope_size/2. - 0.)  # slope bottom
    x4, y4 = -slope_size/2., 0. - R / cos(a) - tan(a) * (+slope_size/2. - 0.)  # slope bottom

    torque_pos = 2*pi/4
    x5, y5 = R * cos(torque_pos), R * sin(torque_pos)  # point on disk
    x6, y6 = 1.5*R * cos(torque_pos), 1.5*R * sin(torque_pos)  # point on disk

    circ_a = np.linspace(0., 2*pi, 100)
    circ_x, circ_y = R * np.cos(circ_a), R * np.sin(circ_a)

    orgs = [np.array([x0, y0]), np.array([x5, y5])]
    dsts = [np.array([x1, y1]), np.array([x6, y6])]
    tmp_th = 0. if  v0 > 0. else pi
    tmp = np.array([R * cos(tmp_th + a), -R * sin(tmp_th + a)])
    orgs.append(tmp)
    dsts.append(tmp)
    oms = [om0, -C]
    rads = [L/2, 1.25*R]
    
    data_xmin, data_xmax = min(x0, x1, x2, x3), max(y0, x1, x2, x3)
    data_ymin, data_ymax = min(x0, y1, y2, y3), max(y0, y1, y2, y3)
    icon_bounds = [data_xmin, data_xmax, data_ymin, data_ymax]
    scl_x, scl_y, pad_x, pad_y = compute_scaling(ax, icon_bounds, fx, fy, pad)

    args = [ax, data_xmax, data_ymin, scl_x, scl_y, pad_x, pad_y]
    ox, oy = map_xy(0., 0., *args)
    x1, y1 = map_xy(x1, y1, *args)
    x2, y2 = map_xy(x2, y2, *args)
    x3, y3 = map_xy(x3, y3, *args)
    x4, y4 = map_xy(x4, y4, *args)
    x5, y5 = map_xy(x5, y5, *args)
    x6, y6 = map_xy(x6, y6, *args)
    circ_x, circ_y = map_xy(circ_x, circ_y, *args)
    
    c2, c3, c4 = get_color_palette(c1)
    ms = 15 * (0.1*max(w_disp, h_disp))
    LW = lw * (0.1*max(w_disp, h_disp))
    ax.fill(circ_x, circ_y, color=c2, edgecolor='none', zorder=0, alpha=0.35) # disk
    ax.plot(
        [ox], [oy], 'o', ms=ms*np.sqrt(0.5), zorder=1, 
        markerfacecolor=c3, markeredgecolor="none"
    ) # center
    ax.plot([x1], [y1], 'o', ms=ms*np.sqrt(m/(m+M)), color=c1, zorder=1) # node 1
    ax.plot([ox, x1], [oy, y1], ls='-', lw=LW, color=c2, zorder=0) # rod
    ax.plot([x2, x3], [y2, y3], ls='-', lw=LW, color=c3, zorder=0) # slope
    # ax.fill([x2, x3, x4], [y2, y3, y4], ls='-', lw=2, color=color, alpha=0.5, edgecolor='none') # slope
    
    for i, (org, dst, om, rad) in enumerate(zip(orgs, dsts, oms, rads)):
        strength = 0.10*om
        arrow_pts, arrow_head = get_curved_arrow(org, dst, rad, strength, head_pct=0.20)
        a_head_x, a_head_y = map_xy(arrow_head[:, 0], arrow_head[:, 1], *args)
        arrow_pts_x, arrow_pts_y = map_xy(arrow_pts[:, 0], arrow_pts[:, 1], *args)
        ax.plot(arrow_pts_x, arrow_pts_y, ls='-', color=c4)
        ax.fill(a_head_x, a_head_y, color=c4, edgecolor='none')
        
    return

def draw_icon(
    ax, name, sim, fraction_x=0.15, fraction_y=0.15, 
    pad=0.02, c_light="lightgrey", lw=2
    ):
    """
    fraction indicates the maximum width of the logo as a % of full figure width
    """
    # fix the bounds
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    if name == "double pendulum":
        icon_dbl_pendulum(ax, sim, fraction_x, fraction_y, pad, c_light, lw)
    elif name == "triple pendulum":
        icon_dbl_pendulum(ax, sim, fraction_x, fraction_y, pad, c_light, lw)
    elif name == "driven pendulum":
        icon_dbl_pendulum(ax, sim, fraction_x, fraction_y, pad, c_light, lw, driven=True)
    elif name == "atwood":
        icon_atwood(ax, sim, fraction_x, fraction_y, pad, c_light, lw)
    elif name == "vertical pendulum":
        icon_mv_pendulum(ax, sim, fraction_x, fraction_y, pad, c_light, lw, vertical=True)
    elif name == "horizontal pendulum":
        icon_mv_pendulum(ax, sim, fraction_x, fraction_y, pad, c_light, lw, vertical=False)
    elif name == "cylinder":
        icon_cylinder(ax, sim, fraction_x, fraction_y, pad, c_light, lw)

    return


if __name__ == "__main__":

    setup = {"t_sim": 30., "fps": 30., "slowdown": 1., "oversample": 5}

    fig, ax = plt.subplots(1, 1, figsize=(10., 10.), constrained_layout=False)
    fig.tight_layout()
    ax.plot([-1., 1000.], [0., 3.], '-o')

    prm = {'g': 9.81, 'l1': 1.0, 'l2': 2.0, 'm1': 2.0, 'm2': 3.}
    initials = {'phi1': 1, 'phi2': 1-pi/2, 'om1': 2.,'om2': 3.}
    sim = DoublePendulum(setup, prm, initials)
    draw_icon(ax, "double pendulum", sim, 0.50, 0.25, 0.04)

    # prm = {'g': 9.81, 'l1': 1.0, 'l2': 2.0, 'm1': 2.0, 'm2': 3.}
    # initials = {'phi1': 1, 'phi2': 1-pi/2, 'om1': 2.,'om2': 3.}
    # sim = DrivenPendulum(setup, prm, initials)
    # draw_icon(ax, "driven pendulum", sim, 0.50, 0.25, 0.04)

    # prm = {'g': 9.81, 'l1': 1.0, 'l2': 2.0, 'l3': 2.0, 'm1': 2.0, 'm2': 3., 'm3':2.5}
    # initials = {'phi1': 1, 'phi2': 1-pi/2, 'phi3': -pi/4, 'om1': 2.,'om2': 3., 'om3': 0}
    # sim = TriplePendulum(setup, prm, initials)
    # draw_icon(ax, "triple pendulum", sim, 0.25, 0.25, 0.04)

    # prm = {'g': 9.81, 'M': 5.0, 'm': 4.0}
    # initials = {'r': 0.75, 'dr': 0.0, 'th': -pi/4, 'om': 1.}
    # sim = AtwoodPendulum(setup, prm, initials)
    # draw_icon(ax, "atwood", sim, 0.25, 0.25, 0.04)

    # prm = {'g': 9.81, 'l': 1.0, 'mb': 5.0, 'mp': 4.0, 'F': 1.0, 'w': 3.}
    # initials = {'phi': 0.75, 'om': 3.0, 'x': 0., 'dx': 0.4}
    # sim = VerticalPendulum(setup, prm, initials)
    # draw_icon(ax, "vertical pendulum", sim, 0.25, 0.25, 0.04)

    # prm = {
    #     'C': -15.65, 'alpha': np.radians(10), 'R': 0.8, 'M': 9.5, 'm': 1.0, 'L': 1.,
    #     'g': 9.81, 'D1': 0., 'D2': 0.
    # }
    # initials = {'th': np.radians(170), 'om': 0.0, 'x': 0., 'dx': 1.0}
    # sim = Cylinder(setup, prm, initials)
    # draw_icon(ax, "cylinder", sim, 0.25, 0.25, 0.04)


    plt.show()

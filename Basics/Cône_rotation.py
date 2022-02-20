import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, tan, arcsin, arctan

plt.rcParams['legend.fontsize'] = 10

if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')

    # Prepare arrays x, y, z
    n, L = 2, 4
    R = L * tan(arcsin(1 / n))
    alpha = arctan(R / L)
    theta = np.linspace(0, 2 * pi, 1000)
    sa, ca = sin(alpha), cos(alpha)
    st, ct = sin(theta), cos(theta)
    phi = -theta / sa
    sp, cp = sin(phi), cos(phi)

    x = -L * st * ca + R * (ct * cp - sa * st * sp)
    y = L * ca * ct + R * (st * cp + sa * ct * sp)
    z = L * sa - R * ca * sp

    ax.set_xlim3d(-5 / 4 * L, 5 / 4 * L)
    ax.set_ylim3d(-5 / 4 * L, 5 / 4 * L)
    ax.set_zlim3d(-R / 5, 9 / 4 * R)
    ax.plot(x, y, z, label='parametric curve')
    ax.legend()

    plt.show()

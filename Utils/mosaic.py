import sys
import os
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
from Atwood_machine.atwood_pendulum import Atwood_Simulation, atwood_ode, atwood_kinematics
from Cylinder_fall.falling_cylinder import Cylinder_Simulation, cylinder_ode, cylinder_kinematics  # not done yet
from Pendulum_2.double_pendulum import ...
from Pendulum_2.double_pendulum import ...
from Pendulum_3.triple_pendulum import TriplePendulum, triple_pendulum_ode, triple_pendulum_kinematics



# Draw
setup = {"t_sim": tsim, "fps": 30., "slowdown": 1., "oversample": 15}
simulation = Atwood_Simulation(params, initial, setup)
time_series = atwood_ode(simulation)
x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = atwood_kinematics(simulation, time_series)
see_path(1., [array([x2, y2]), array([2*x2, 2*y2])], [v, v], ["Blues", "viridis"], var_case=2, save="no")


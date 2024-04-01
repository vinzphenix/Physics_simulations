import matplotlib.pyplot as plt
import numpy as np
from physicsim.pendulum_atwood import AtwoodPendulum
from physicsim.cylinder_slide import Cylinder
from physicsim.pendulum_driven import DrivenPendulum
from physicsim.pendulum_2 import DoublePendulum
from physicsim.pendulum_3 import TriplePendulum
from physicsim.pendulum_elastic import PendulumElastic
from physicsim.pendulum_vertical import VerticalPendulum
from physicsim.pendulum_horiztontal import HorizontalPendulum

from scripts.atwood.run_atwood import load_configuration as atwood_config
from scripts.cylinder_slide.run_cylinder import load_configuration as cylinder_config
from scripts.pendulum_driven.run_pendulum_driven import load_configuration as driven_config
from scripts.pendulum_2.run_pendulum2 import load_configuration as double_config
from scripts.pendulum_3.run_pendulum3 import load_configuration as triple_config
from scripts.pendulum_elastic.run_pendulum_elastic import load_configuration as elastic_config
from scripts.pendulum_inverted.run_pendulum_v import load_configuration as vertical_config
from scripts.pendulum_inverted.run_pendulum_h import load_configuration as horizontal_config

from utils import icon
import os
import sys
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

from utils import icon, display

setup = {"t_sim": 30., "fps": 30., "slowdown": 1., "oversample": 10}

def generate_figures(i=0):
    
    # Cylinder slide
    if i in [0, 1]:
        setup["t_sim"] = 64.
        prm, initials = cylinder_config(1)
        sim = Cylinder(setup, prm, initials)
        sim.solve_ode()
        th, om, x, dx = sim.full_series
        s = np.s_[:245:-1]
        display.see_path(
            om[s], dx[s], th[s], colors='Blues', lws=2.,
            var_case=2, figsize=(10., 10.),
            save="./Mosaic/figure_18.png", displayedInfo="", 
            icon_name="cylinder", sim=sim
        )

    # Atwood pendulum
    if i in [0, 2]:
        setup["t_sim"] = 99.7
        prm, initials = atwood_config(3)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        display.see_path(
            om, -r, dr, colors='magma', lws=2.,
            var_case=2, figsize=(10., 10.),
            save="./Mosaic/figure_13.png", displayedInfo="", 
            icon_name="atwood", sim=sim
        )

    # Atwood pendulum
    if i in [0, 3]:
        setup["t_sim"] = 150.
        prm, initials = atwood_config(10)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        display.see_path(
            th, dr, r, colors='inferno', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.1, 0.65),
            save="./Mosaic/figure_03.png", displayedInfo="", 
            icon_name="atwood", sim=sim
        )

    # Atwood pendulum
    if i in [0, 4]:
        setup["t_sim"] = 150.
        prm, initials = atwood_config(11)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics
        display.see_path(
            om, v, np.abs(om), colors='Blues', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.1, 0.5),
            save="./Mosaic/figure_04.png", displayedInfo="", 
            icon_name="atwood", sim=sim
        )

    # # Atwood pendulum
    # if i in [0, 5]:
    #     setup["t_sim"] = 150.
    #     prm, initials = atwood_config(52)
    #     sim = AtwoodPendulum(setup, prm, initials)
    #     sim.solve_ode()
    #     r, dr, th, om = sim.full_series
    #     x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics
    #     display.see_path(
    #         np.cos(th)**2, r, v, colors='Blues', lws=1.,
    #         var_case=2, figsize=(10., 10.), pad=(0.1, 0.2),
    #         save="./Mosaic/figure_100.png", displayedInfo="", 
    #         icon_name="atwood", sim=sim
    #     )
    
    # Atwood pendulum
    if i in [0, 6]:
        setup["t_sim"] = 755.
        # setup["oversample"] = 20
        prm, initials = atwood_config(55)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics
        display.see_path(
            x2, y2, v, colors='jet', lws=0.5,
            var_case=2, figsize=(10., 10.), pad=(0.1, 0.15),
            save="./Mosaic/figure_12.png", displayedInfo="", 
            icon_name="atwood", sim=sim
        )
    
    # Double pendulum
    if i in [0, 7]:
        setup["t_sim"] = 500.
        prm, initials = double_config(3)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1*om1, om2, v1, colors='inferno_r', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.12, 0.10),
            save="./Mosaic/figure_01.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 8]:
        setup["t_sim"] = 501.
        prm, initials = double_config(4)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1, phi2*om1, phi2, colors='Reds', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.12, 0.10),
            save="./Mosaic/figure_17.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 9]:
        setup["t_sim"] = 750.
        prm, initials = double_config(22)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1*om2, phi2*om1, v1*v2, colors='Reds', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.20, 0.15),
            save="./Mosaic/figure_06.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 10]:
        setup["t_sim"] = 500.
        prm, initials = double_config(24)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            om1, om2, v2, colors='Blues', lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.15, 0.15),
            save="./Mosaic/figure_11.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )
    
    # Double pendulum
    if i in [0, 11]:
        setup["t_sim"] = 500.
        prm, initials = double_config(42)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            [phi1, phi2], [om2, om1], [v1, v2], 
            colors="inferno", lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.15, 0.15),
            save="./Mosaic/figure_08.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 12]:
        setup["t_sim"] = 500.
        prm, initials = double_config(45)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            [phi1, phi2], [om1, om2], [phi2, phi1], 
            colors=["Reds", "Reds_r"], lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.15, 0.15),
            save="./Mosaic/figure_10.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 13]:
        setup["t_sim"] = 400.
        prm, initials = double_config(45)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            [4*om1*phi2, phi1], 
            [5  *om2**2-0.5, 1.25*np.abs(-0.75+np.abs(om2+1))],
            [om2*phi1, phi2], 
            colors=["turbo_r", "turbo"], lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.25, 0.25, 0.35, 0.15),
            save="./Mosaic/figure_05.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Double pendulum
    if i in [0, 14]:
        setup["t_sim"] = 300.
        prm, initials = double_config(55)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2, a2 = sim.full_kinematics[:7]
        display.see_path(
            x2, y2, v2, 
            colors="jet", lws=1.,
            var_case=1, figsize=(10., 10.), pad=(0.15, 0.15),
            save="./Mosaic/figure_02.png", displayedInfo="", 
            icon_name="double pendulum", sim=sim
        )

    # Driven pendulum
    if i in [0, 15]:
        setup["t_sim"] = 59.
        prm, initials = driven_config(5)
        sim = DrivenPendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, x2, y2, v2, a2 = sim.full_kinematics[:6]
        display.see_path(
            x2, y2, v2, 
            colors="jet", lws=1.,
            var_case=1, figsize=(10., 10.), pad=(0.05, 0.15),
            save="./Mosaic/figure_15.png", displayedInfo="", 
            icon_name="driven pendulum", sim=sim
        )

    # Driven pendulum
    if i in [0, 16]:
        setup["t_sim"] = 2000.
        prm, initials = driven_config(11)
        sim = DrivenPendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, x2, y2, v2, a2 = sim.full_kinematics[:6]
        display.see_path(
            x2, y2, v2, 
            colors="Blues", lws=1.,
            var_case=1, figsize=(10., 10.), pad=(0.15, 0.15),
            save="./Mosaic/figure_14.png", displayedInfo="", 
            icon_name="driven pendulum", sim=sim
        )

    # Horizontal pendulum
    if i in [0, 17]:
        setup["t_sim"] = 30.
        prm, initials = horizontal_config(7)
        sim = HorizontalPendulum(setup, prm, initials)
        sim.solve_ode()
        x, dx, th, om = sim.full_series
        xb, yb, xp, yp, vp = sim.full_kinematics[:5]
        display.see_path(
            -om, dx, vp, 
            colors="Spectral", lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.15, 0.20),
            save="./Mosaic/figure_07.png", displayedInfo="", 
            icon_name="horizontal pendulum", sim=sim
        )

    # Horizontal pendulum
    if i in [0, 18]:
        setup["t_sim"] = 200.
        prm, initials = horizontal_config(9)
        sim = HorizontalPendulum(setup, prm, initials)
        sim.solve_ode()
        x, dx, th, om = sim.full_series
        xb, yb, xp, yp, vp = sim.full_kinematics[:5]
        display.see_path(
            th, dx, np.abs(om), 
            colors="Blues_r", lws=1.,
            var_case=2, figsize=(10., 10.), pad=(0.15, 0.20),
            save="./Mosaic/figure_09.png", displayedInfo="", 
            icon_name="horizontal pendulum", sim=sim
        )

    # Triple pendulum
    if i in [0, 19]:
        setup["t_sim"] = 7.
        setup["oversample"] = 20
        prm, initials = triple_config(3)
        sim = TriplePendulum(setup, prm, initials)
        setup["oversample"] = 10
        sim.solve_ode()
        x3, y3, v3 = sim.full_kinematics[[4, 5, 11]]
        display.see_path(
            x3, y3, v3, 
            colors="jet", lws=3.,
            var_case=1, figsize=(10., 10.), pad=(0.15, 0.20),
            save="./Mosaic/figure_16.png", displayedInfo="", 
            icon_name="triple pendulum", sim=sim
        )

    return

def merge_images():

    working_dir = f"./Figures/Mosaic/"

    files = [
        os.path.join(working_dir, file) 
        for file in os.listdir(working_dir) 
        if file.startswith("figure_") and file.endswith(".png")
    ]

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    n_images = len(files)
    n_rows, n_cols = 3, 6
    
    if n_cols * n_rows != n_images:
        raise ValueError("Number of images does not match grid size")

    images = [Image.open(x) for x in files]
    widths = np.array([i.size[0] for i in images])
    heights = np.array([i.size[1] for i in images])

    if np.any(widths != widths[0]) or np.any(heights != heights[0]):
        raise ValueError("Only same sizes images currently supported")

    pad_out = 50
    pad_in = 10

    total_width = n_cols * widths[0] + (n_cols - 1) * pad_in + 2 * pad_out
    total_height = n_rows * heights[0] + (n_rows - 1) * pad_in + 2 * pad_out

    new_im = Image.new('RGBA', (total_width, total_height), color=(255,255,255))

    for i in range(n_images):
        row = i // n_cols
        col = i % n_cols
        x, y = pad_out + col * (widths[0] + pad_in), pad_out + row * (heights[0] + pad_in)
        new_im.paste(images[i], (x, y))

    new_im.save(f"{working_dir}mosaic.png")
    return


if __name__ == "__main__":
    generate_figures(0)
    merge_images()

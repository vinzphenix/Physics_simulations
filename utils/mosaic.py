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
from scripts.cylinder_slide.run_cylinder import \
    load_configuration as cylinder_config
from scripts.pendulum_driven.run_pendulum_driven import \
    load_configuration as driven_config
from scripts.pendulum_2.run_pendulum2 import \
    load_configuration as double_config
from scripts.pendulum_3.run_pendulum3 import \
    load_configuration as triple_config
from scripts.pendulum_elastic.run_pendulum_elastic import \
    load_configuration as elastic_config
from scripts.pendulum_inverted.run_pendulum_v import \
    load_configuration as vertical_config
from scripts.pendulum_inverted.run_pendulum_h import \
    load_configuration as horizontal_config

from utils import icon, display

EXTENSION = "png"
setup = {"t_sim": 30., "fps": 30., "slowdown": 1., "oversample": 10}

def generate_figures(
    i=0, figsize=(10., 10.), bg="gradient", 
    icon_size=(0.17, 0.17), save=False
):
    
    scale = 1.0 * (figsize[0] / 15.)**2
    
    # Cylinder slide
    filename = f"./Mosaic/figure_02.{EXTENSION:s}" if save else ""
    if i in [0, 1]:
        setup["t_sim"] = 64.
        prm, initials = cylinder_config(1)
        sim = Cylinder(setup, prm, initials)
        sim.solve_ode()
        th, om, x, dx = sim.full_series
        s = np.s_[:245:-1]
        display.see_path(
            om[s], dx[s], th[s], colors='Blues', lws=2.*scale,
            var_case=2, figsize=figsize,
            save=filename, displayedInfo="", bg=bg,
            icon_name="cylinder", sim=sim, 
            icon_size=(1.30*icon_size[0], icon_size[1])
        )

    # Atwood pendulum
    filename = f"./Mosaic/figure_13.{EXTENSION:s}" if save else ""
    if i in [0, 2]:
        setup["t_sim"] = 99.7
        prm, initials = atwood_config(3)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        display.see_path(
            om, -r, dr, colors='inferno', lws=2.*scale,  # magma
            var_case=2, figsize=figsize,
            save=filename, displayedInfo="", bg=bg,
            icon_name="atwood", sim=sim, icon_size=icon_size
        )

    # Atwood pendulum
    filename = f"./Mosaic/figure_18.{EXTENSION:s}" if save else ""
    if i in [0, 3]:
        setup["t_sim"] = 150.
        prm, initials = atwood_config(10)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        display.see_path(
            th, dr, r, colors='inferno', lws=1.*scale,  # inferno
            var_case=2, figsize=figsize, pad=(0.1, 0.65),
            save=filename, displayedInfo="", bg=bg,
            icon_name="atwood", sim=sim, icon_size=icon_size
        )

    # Atwood pendulum
    filename = f"./Mosaic/figure_11.{EXTENSION:s}" if save else ""
    if i in [0, 4]:
        setup["t_sim"] = 150.
        prm, initials = atwood_config(11)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics
        display.see_path(
            om, v, np.abs(om), colors='Blues', lws=1.*scale,
            var_case=2, figsize=figsize, pad=(0.1, 0.5),
            save=filename, displayedInfo="", bg=bg,
            icon_name="atwood", sim=sim, icon_size=icon_size
        )
    
    # Atwood pendulum
    filename = f"./Mosaic/figure_09.{EXTENSION:s}" if save else ""
    if i in [0, 6]:
        setup["t_sim"] = 755.
        # setup["oversample"] = 20
        prm, initials = atwood_config(55)
        sim = AtwoodPendulum(setup, prm, initials)
        sim.solve_ode()
        r, dr, th, om = sim.full_series
        x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics
        display.see_path(
            x2, y2, v, colors='Blues_r', lws=0.5*scale,  # jet
            var_case=2, figsize=figsize, pad=(0.1, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="atwood", sim=sim, icon_size=icon_size
        )
    
    # Double pendulum
    filename = f"./Mosaic/figure_01.{EXTENSION:s}" if save else ""
    if i in [0, 7]:
        setup["t_sim"] = 500.
        prm, initials = double_config(3)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1*om1, om2, v1, colors='inferno_r', lws=1.*scale, # inferno_r
            var_case=2, figsize=figsize, pad=(0.12, 0.10),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_17.{EXTENSION:s}" if save else ""
    if i in [0, 8]:
        setup["t_sim"] = 501.
        prm, initials = double_config(4)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1, phi2*om1, phi2, colors='Reds', lws=1.*scale,
            var_case=2, figsize=figsize, pad=(0.14, 0.10),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_05.{EXTENSION:s}" if save else ""
    if i in [0, 9]:
        setup["t_sim"] = 750.
        prm, initials = double_config(22)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            phi1*om2, phi2*om1, v1*v2, colors='Reds', lws=1.*scale,
            var_case=2, figsize=figsize, pad=(0.20, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_08.{EXTENSION:s}" if save else ""
    if i in [0, 10]:
        setup["t_sim"] = 500.
        prm, initials = double_config(24)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            om1, om2, v2, colors='Reds', lws=2.*scale,
            var_case=2, figsize=figsize, pad=(0.15, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )
    
    # Double pendulum
    filename = f"./Mosaic/figure_06.{EXTENSION:s}" if save else ""
    if i in [0, 11]:
        setup["t_sim"] = 500.
        prm, initials = double_config(42)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            [phi1, phi2], [om2, om1], [v1, v2], 
            colors="inferno", lws=1.*scale,  # inferno
            var_case=2, figsize=figsize, pad=(0.15, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_10.{EXTENSION:s}" if save else ""
    if i in [0, 12]:
        setup["t_sim"] = 500.
        prm, initials = double_config(45)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2 = sim.full_kinematics[:6]
        display.see_path(
            [phi1, phi2], [om1, om2], [phi2, phi1], 
            colors=["Reds", "Reds_r"], lws=1.*scale,
            var_case=2, figsize=figsize, pad=(0.15, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_03.{EXTENSION:s}" if save else ""
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
            colors=["turbo_r", "turbo"], lws=1.*scale,  # turbo_r, turbo
            var_case=2, figsize=figsize, pad=(0.25, 0.25, 0.30, 0.20),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Double pendulum
    filename = f"./Mosaic/figure_04.{EXTENSION:s}" if save else ""
    if i in [0, 14]:
        setup["t_sim"] = 300.
        prm, initials = double_config(55)
        sim = DoublePendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, v1, x2, y2, v2, a2 = sim.full_kinematics[:7]
        display.see_path(
            x2, y2, v2, 
            colors="turbo", lws=1.*scale,  # jet
            var_case=1, figsize=figsize, pad=(0.15, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="double pendulum", sim=sim, icon_size=icon_size
        )

    # Driven pendulum
    filename = f"./Mosaic/figure_15.{EXTENSION:s}" if save else ""
    if i in [0, 15]:
        setup["t_sim"] = 59.
        prm, initials = driven_config(5)
        sim = DrivenPendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, x2, y2, v2, a2 = sim.full_kinematics[:6]
        display.see_path(
            x2, y2, v2, 
            colors="jet", lws=2.0*scale,  # jet
            var_case=1, figsize=figsize, pad=(0.05, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="driven pendulum", sim=sim, icon_size=icon_size
        )

    # Driven pendulum
    filename = f"./Mosaic/figure_14.{EXTENSION:s}" if save else ""
    if i in [0, 16]:
        setup["t_sim"] = 2000.
        prm, initials = driven_config(11)
        sim = DrivenPendulum(setup, prm, initials)
        sim.solve_ode()
        phi1, om1, phi2, om2 = sim.full_series
        x1, y1, x2, y2, v2, a2 = sim.full_kinematics[:6]
        display.see_path(
            x2, y2, v2, 
            colors="Blues", lws=1.*scale,
            var_case=1, figsize=figsize, pad=(0.15, 0.15),
            save=filename, displayedInfo="", bg=bg,
            icon_name="driven pendulum", sim=sim, icon_size=icon_size
        )

    # Horizontal pendulum
    filename = f"./Mosaic/figure_12.{EXTENSION:s}" if save else ""
    if i in [0, 17]:
        setup["t_sim"] = 30.
        prm, initials = horizontal_config(7)
        sim = HorizontalPendulum(setup, prm, initials)
        sim.solve_ode()
        x, dx, th, om = sim.full_series
        xb, yb, xp, yp, vp = sim.full_kinematics[:5]
        display.see_path(
            -om, dx, vp, 
            colors="Reds", lws=1.*scale,  # Spectral
            var_case=2, figsize=figsize, pad=(0.15, 0.20),
            save=filename, displayedInfo="", bg=bg,
            icon_name="horizontal pendulum", sim=sim, icon_size=icon_size
        )

    # Horizontal pendulum
    filename = f"./Mosaic/figure_07.{EXTENSION:s}" if save else ""
    if i in [0, 18]:
        setup["t_sim"] = 200.
        prm, initials = horizontal_config(9)
        sim = HorizontalPendulum(setup, prm, initials)
        sim.solve_ode()
        x, dx, th, om = sim.full_series
        xb, yb, xp, yp, vp = sim.full_kinematics[:5]
        display.see_path(
            th, dx, np.abs(om), 
            colors="Blues_r", lws=1.*scale,
            var_case=2, figsize=figsize, pad=(0.15, 0.20),
            save=filename, displayedInfo="", bg=bg,
            icon_name="horizontal pendulum", sim=sim, icon_size=icon_size
        )

    # Triple pendulum
    filename = f"./Mosaic/figure_16.{EXTENSION:s}" if save else ""
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
            colors="jet", lws=4.*scale,  # jet
            var_case=1, figsize=figsize, pad=(0.15, 0.20),
            save=filename, displayedInfo="", bg=bg,
            icon_name="triple pendulum", sim=sim, icon_size=icon_size
        )
    
    return

def merge_images(n_rows, n_cols, W, H, pad_out, pad_in):

    working_dir = f"./Figures/Mosaic/"

    files = [
        os.path.join(working_dir, file) 
        for file in os.listdir(working_dir) 
        if file.startswith("figure_") and file.endswith(f".{EXTENSION:s}")
    ]

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    n_images = len(files)
    
    if n_cols * n_rows != n_images:
        raise ValueError("Number of images does not match grid size")

    images = [Image.open(x) for x in files]
    widths = np.array([i.size[0] for i in images])
    heights = np.array([i.size[1] for i in images])

    if np.any(widths != widths[0]) or np.any(heights != heights[0]):
        raise ValueError("Only same sizes images currently supported")
        
    pad_out_px = int(np.ceil(pad_out * W))
    pad_in_px = int(np.ceil(pad_in * W))
    data_w = W - 2 * pad_out_px - (n_cols - 1) * pad_in_px
    data_h = H - 2 * pad_out_px - (n_rows - 1) * pad_in_px
    ref_diff_w = diff_w = n_cols * widths[0] - data_w
    ref_diff_h = diff_h = n_rows * heights[0] - data_h
    
    # Crop left and top sides of the images by some pixels
    # to reach the desired total width and height
    print(f"Cropped width  = {100*diff_w/widths[0]:.2f}% of the original")
    print(f"Cropped height = {100*diff_h/heights[0]:.2f}% of the original")
    
    new_im = Image.new('RGB', (W, H), color=(255,255,255))
    
    i = 0
    y = pad_out_px
    diff_h = ref_diff_h
    for row in range(n_rows):
        x = pad_out_px
        diff_w = ref_diff_w
        delta_y = diff_h // (n_rows - row)
        diff_h -= delta_y
        for col in range(n_cols):
            delta_x = diff_w // (n_cols - col)
            diff_w -= delta_x
            width, height = images[i].size
            image = images[i].crop((delta_x, delta_y, width, height))
            new_im.paste(image, (x, y))
            print(f"Image {i+1:02d} with size : {image.size}")
            x += width - delta_x + pad_in_px
            i += 1
        y += height - delta_y + pad_in_px
    
    new_im.save(f"{working_dir}mosaic.{EXTENSION:s}")
    return


def create_mosaic():
    n_rows = 3
    n_cols = 6
    aspect = 2.
    W = 15000
    H = int(np.round(W / aspect))
    pad_out = 0.015
    pad_in = 0.0015
    # (c-1)*pad_in*W + 2*pad_out*W + c*w = W
    # (r-1)*pad_in*W + 2*pad_out*W + r*h = W / aspect
    w = W*(1. - (n_cols - 1) * pad_in - 2 * pad_out) / n_cols
    h = W*(1./aspect - (n_rows - 1) * pad_in - 2 * pad_out) / n_rows
    w = int(np.ceil(w))
    h = int(np.ceil(h))
    
    generate_figures(0, figsize=(w/100., h/100.), bg=True, save=True)
    merge_images(n_rows, n_cols, W, H, pad_out, pad_in)
    return


if __name__ == "__main__":
    
    # generate_figures(13, figsize=(20., 20.), bg=True, save=False)
    # generate_figures(6, figsize=(20., 20.), bg=True, save=False)
    # generate_figures(10, figsize=(20., 20.), bg=True, save=False)
    # generate_figures(1, figsize=(10., 10.), bg=True)
    # generate_figures(12, figsize=(10., 10.), bg=True, save=False)
    
    create_mosaic()
    pass
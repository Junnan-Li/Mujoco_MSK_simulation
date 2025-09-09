import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation
from src.visualizer import MusculoskeletalVisualizer
# from src.IKParams import IK_Params, IK_Target
# from src.IK_Solver import IK_Solver, IK_Algorithm
# import matplotlib.pyplot as plt
import mujoco
import time
import itertools
from scipy.spatial.transform import Rotation as R

# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/hand/myohand_markers.xml')

viz = MusculoskeletalVisualizer(sim, azimuth=180, elevation=0,distance=0.1,lookat=[0.4, -0.25, 1.5])


body_names = ["secondmc"]

duration = 5
with mujoco.viewer.launch_passive(
            sim.model, 
            sim.data,
            show_left_ui=False,show_right_ui=False
        ) as viz.viewer:
        viz._viewer_settings()
            
        # viz.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        # viz.viewer.sync()

        start_time = time.time()
        sim_start_time = viz.sim.data.time
        last_log_time = 0
        
        viz.render()
        # print(viz.viewer.opt.flags)
        while viz.viewer.is_running() and (viz.sim.data.time - sim_start_time) < duration:
            # Get control input


            viz.viewer.user_scn.ngeom = 0

            viz.draw_body_frame(body_names=body_names)
            # viz.draw_site_frame(site_names=['IFtip', 'MFtip','RFtip'])

 
            viz.render()

            elapsed = time.time() - start_time
            sim_time = sim.data.time - sim_start_time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)




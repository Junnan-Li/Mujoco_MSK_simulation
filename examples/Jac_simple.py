import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer
# import matplotlib.pyplot as plt
import mujoco
import time
import itertools



# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/serial_robot.xml')

# Set muscle control mode
muscle_params = {
    'activation_dynamics': False,
    'tau_activation': 0.015,
    'tau_deactivation': 0.060
}
sim.set_control_mode(ControlMode.MUSCLE, muscle_params)


viz = MusculoskeletalVisualizer(sim, azimuth=90, elevation=0, distance=1,lookat=[0, -0.5, 0.5])
# viz.run_simulation(muscle_activation_pattern,log_function=log_data_runtime , duration=5.0, log_interval=1.0)

print(f'{sim.joint_names}')
# plot


# joint_names =  ['mcp2_flexion','mcp2_abduction','pm2_flexion','md2_flexion']

# index = [i for i in range(38) if i < 10 or i > 17]
# jnt_lock_names = []
# for i in index:
#     jnt_lock_names.append(sim.joint_names[i])
# jnt_lock_values = np.zeros(len(jnt_lock_names))
# sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)

sim.model.vis.scale.framelength = 0.2
sim.model.vis.scale.framewidth = 0.03
sim.model.vis.scale.jointlength = 1
sim.model.vis.scale.jointwidth = 0.015
sim.integrate = True
duration = 5
with mujoco.viewer.launch_passive(
            sim.model, 
            sim.data,
            show_left_ui=False,show_right_ui=False
        ) as viz.viewer:

        viz._viewer_settings()
        viz.render()
        # viz.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
        # viz.viewer.sync()

        start_time = time.time()
        sim_start_time = viz.sim.data.time
        last_log_time = 0
        
        # print(viz.viewer.opt.flags)
        while viz.viewer.is_running() and (viz.sim.data.time - sim_start_time) < duration:
            # Get control input
            current_time = viz.sim.data.time

            current_pose = viz.sim.data.qpos.copy() 
            # viz.sim.data.qpos = np.clip(current_pose + (np.random.rand(len(viz.sim.data.qpos))-0.5)*np.pi /180,
            #                             sim.model.jnt_range[:,0],sim.model.jnt_range[:,1])

            sim.step(np.zeros(sim.model.nu))

            viz.viewer.user_scn.ngeom = 0
            viz.draw_site_frame(site_names=['site1'],AxisLen=0.2)

            jac = sim.get_site_Jac('site1')
            # viz.viewer.user_scn.ngeom = i
            # print(viz.viewer.user_scn.ngeom)

            # Render
            viz.render()
            # print(sim.model.jnt_range[:3])
            # print(sim.data.qpos[:3])

            elapsed = time.time() - start_time
            sim_time = sim.data.time - sim_start_time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)



# plt.figure(figsize=(10, 5))
# for i in range(sim.record_data["qpos"].shape[1]):
#     plt.plot(sim.record_data['time'], sim.record_data['ctrl'][:, i], label=f'ctrl[{i}]')
#     plt.plot(sim.record_data['time'], sim.record_data['act'][:, i], label=f'act[{i}]')
# plt.xlabel('Time [s]')
# plt.ylabel('Joint Position')
# plt.title('Joint Position over Time')
# plt.legend()
# plt.grid(True)
# # plt.tight_layout()
# plt.show()


# viz.close()
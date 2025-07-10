import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer
# import matplotlib.pyplot as plt
import mujoco
import time


# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/arm/myoarm.xml')

# Set muscle control mode
muscle_params = {
    'activation_dynamics': False,
    'tau_activation': 0.015,
    'tau_deactivation': 0.060
}
sim.set_control_mode(ControlMode.MUSCLE, muscle_params)


viz = MusculoskeletalVisualizer(sim, azimuth=180, elevation=0, distance=1,lookat=[0.4, -0.25, 1.5])

# jnt_lock_names = ['pro_sup','flexion','mcp2_abduction','pm2_flexion']
# jnt_lock_values = np.array([0,0,0,0.3])
# sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)

# viz.run_simulation(muscle_activation_pattern,log_function=log_data_runtime , duration=5.0, log_interval=1.0)

# print(f'{sim.record_data['time'].shape[0]}')
# plot

MSK_state = sim.get_musculoskeletal_state()

muscle_activations = sim.get_muscle_activations()
muscle_forces = sim.get_muscle_forces()
qpos = sim.get_joint_positions()
qvel = sim.get_joint_velocities()
qtorque = sim.get_joint_torques()


duration = 5
with mujoco.viewer.launch_passive(
            sim.model, 
            sim.data,
            show_left_ui=False,show_right_ui=False
        ) as viz.viewer:
        viz._viewer_settings()
            
        start_time = time.time()
        sim_start_time = viz.sim.data.time
        last_log_time = 0
        
        print(viz.viewer.opt.flags)
        while viz.viewer.is_running() and (viz.sim.data.time - sim_start_time) < duration:
            # Get control input
            current_time = viz.sim.data.time
        

            # # write to sim.record_
            # record_function(self.sim)

            # Render
            viz.render()



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
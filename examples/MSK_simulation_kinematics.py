# This script simulates the first experiment of the cadaver experiment to identify the 
# kinematic parameters: 
#       CoR and AoR 
#       moment arm curves  
#       (option) joint passive torques
#
# The model is modified from Myohand by add markers and passive joint torques
#

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation
from src.visualizer import MusculoskeletalVisualizer
# from src.IKParams import IK_Params, IK_Target
# from src.IK_Solver import IK_Solver, IK_Algorithm
import matplotlib.pyplot as plt
# import mujoco
# import time
# import itertools
from scipy.spatial.transform import Rotation as R
import src.utilities as ut


# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/hand/myohand_markers.xml')



momentarm = sim.get_muscle_moment_arms_curves(['mcp2_flexion'],['FDS2'] )
q_deg = np.degrees(momentarm[:, 0])
r = momentarm[:, 1]

# # Plot
# plt.figure(figsize=(6, 4))
# plt.plot(q_deg, r, linewidth=2)
# plt.xlabel(f"angle (deg)")
# plt.ylabel(f"FDS2 moment arm (m)")
# plt.title(f"Moment Arm Curve (Jacobian): FDS2 about MCP2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()



viz = MusculoskeletalVisualizer(sim, azimuth=180, elevation=0,distance=0.2,lookat=[0.3, -0.5, 1.5])

# joint to be fixed 
jnt_lock_names = ['pro_sup','flexion','deviation','pm2_flexion','md2_flexion'] #,'mcp2_abduction']
jnt_lock_values = np.array([np.pi/2,0,0,0,0])
sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)

# passive joint torque
jnt_passive = ['mcp2_flexion'] #,'pm2_flexion','md2_flexion']
jnt_passive_index = sim.get_jnt_index(jnt_passive)
# Parameters
A = 1.27
B = 0.031
C = 1.83
D = 0.07
E = 47.39   # neutral for first exponential
F = 58.97   # neutral for second exponential
k = 0.0
# muscle force controller
muscle_names =  ['FDS2','FDP2','EDC2','EIP']
sim.control_act_index = sim.get_muscle_index(muscle_names)
# Define muscle activation pattern
def muscle_force_pattern(t:int, model:MusculoskeletalSimulation):
    """Generate coordinated muscle activation pattern"""
    n_muscles = model.n_actuators
    
    # passive joing torques
    for jnt_pass_index in jnt_passive_index:
        qadr = model.model.jnt_qposadr[jnt_pass_index]
        vadr = model.model.jnt_dofadr[jnt_pass_index]
        q = model.data.qpos[qadr]
        qdot = model.data.qvel[vadr]
        tau_stiff = ut.tau_stiff_DE(q, A, B, C, D, E, F)
        tau_damp = ut.tau_damping(qdot, k)
        model.data.qfrc_applied[vadr] = tau_stiff + tau_damp

    f_MIF = model.get_muscle_MIF() 
    # # Activate muscles in sequence
    init_force = 3.0      # baseline force (N)
    step_interval = 1.5 # seconds per increment
    step_size = 1     # N per increment
    time_inc = 10
    max_force = init_force + time_inc * step_size   # N maximum

    # Create wave-like activation pattern
    f_desired = np.zeros(n_muscles)
    # keep extension muscles stable forces
    f_desired[sim.control_act_index[0:4]] = init_force
    f_desired[sim.control_act_index[2]] = 15

    # Compute current force level
    step_index = int(t // step_interval)
    # Reset when exceeding max_force
    if step_index < time_inc:
        f_desired[sim.control_act_index[0]] = init_force + step_size * step_index
    else:
        f_desired[sim.control_act_index[0]] = max_force - step_size * (step_index-time_inc)


    return np.clip(f_desired, 0, f_MIF)




def log_data_runtime(model:MusculoskeletalSimulation):
    activations = model.get_muscle_activations()
    forces = model.get_muscle_forces()
    
    # Print activation summary
    print(f"Time: {model.data.time:.3f}")
    print(f"muscle activation: {activations}")
    print(f"muscle force: {forces}")
    print("---")

def record_data_runtime(model:MusculoskeletalSimulation):
    if len(model.record_data) == 0:
        model.record_data = {
                "time": [],
                "qpos": [],
                "ctrl": [],
                "mfrc": [],
                "tendon_pos" : [],
                "AppliedF": [],
                "marker_pos":[]
            }
    model.record_data["time"].append(model.data.time)
    model.record_data["qpos"].append(model.data.qpos.copy())
    model.record_data["ctrl"].append(model.data.ctrl.copy())
    model.record_data["tendon_pos"].append(model.data.sensordata[0:4].copy())
    model.record_data["mfrc"].append(-model.data.actuator_force.copy())
    model.record_data["AppliedF"].append(model.data.qfrc_applied.copy())
    model.record_data["marker_pos"].append(model.data.sensordata[8:].copy())


viz.run_simulation(muscle_force_pattern,
                   log_function=log_data_runtime,
                   record_function=record_data_runtime,
                   duration=30.0,
                   log_interval=1.0)

# %%

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(len(muscle_names)):
    axes[0].plot(sim.record_data['qpos'][:, jnt_passive_index[0]], sim.record_data['tendon_pos'][:,i], linestyle='-', label=f'tendon length {muscle_names[i]}')
# axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
# axes[0].set_title("joint pos")
axes[0].set_ylabel("joint pos [rad]")
axes[0].grid(True)
axes[0].legend()
for i in range(len(muscle_names)):
    axes[1].plot(sim.record_data['qpos'][:, jnt_passive_index[0]], sim.record_data['tendon_pos'][:,i],label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel('tendon pos [m] ' )
axes[1].legend()
plt.xlabel('q [rad]')
# plt.ylabel('force')
# plt.legend()
plt.grid(True)
# plt.tight_layout()
# plt.show()


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(jnt_passive_index.shape[0]):
    axes[0].plot(sim.record_data['time'], sim.record_data['qpos'][:, jnt_passive_index[i]], linestyle='-', label=f'tendon length {muscle_names[i]}')
# axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
# axes[0].set_title("joint pos")
axes[0].set_ylabel("joint pos [rad]")
axes[0].grid(True)
axes[0].legend()
for i in range(len(muscle_names)):
    axes[1].plot(sim.record_data['time'], sim.record_data['mfrc'][:,sim.control_act_index[i]],label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel('tendon pos [m] ' )
axes[1].legend()
plt.xlabel('q [rad]')
# plt.ylabel('force')
# plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()


# traj = sim.record_data['marker_pos'].reshape(sim.record_data['marker_pos'].shape[0], 7, 3)
# # Create 3D figure
# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection="3d")
# for i in range(7):
#     ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2], label=f"Marker {i+1}")
#     ax.scatter(traj[0, i, 0], traj[0, i, 1], traj[0, i, 2], marker="o", s=50)  # start point
#     ax.scatter(traj[-1, i, 0], traj[-1, i, 1], traj[-1, i, 2], marker="x", s=50)  # end point

# # Labels
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Marker Trajectories")
# ax.legend()

# plt.show()


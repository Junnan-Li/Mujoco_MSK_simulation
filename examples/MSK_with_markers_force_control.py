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

viz = MusculoskeletalVisualizer(sim, azimuth=0, elevation=0,distance=1,lookat=[0.3, -0.25, 1.5])

# joint to be fixed 
jnt_lock_names = ['pro_sup','flexion','deviation'] #,'mcp2_abduction']
jnt_lock_values = np.array([0,0,0])
sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)

# passive joint torque
jnt_passive = ['mcp2_flexion','pm2_flexion','md2_flexion']
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


    # Create wave-like activation pattern
    f_desired = np.zeros(n_muscles)
    # keep extension muscles stable forces
    f_desired[sim.control_act_index[0:4]] = 0

    f_MIF = model.get_muscle_MIF() 
    # # Activate muscles in sequence
    wave_speed = 0.5  # Hz
    phase = 2 * np.pi *  wave_speed * t
    # f_desired[sim.control_act_index[2]] = np.clip (20 +  18 * np.sin(phase), 0, f_MIF[sim.control_act_index[0]])
        
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
                "AppliedF": []
            }
    model.record_data["time"].append(model.data.time)
    model.record_data["qpos"].append(model.data.qpos.copy())
    model.record_data["ctrl"].append(model.data.ctrl.copy())
    model.record_data["mfrc"].append(-model.data.actuator_force.copy())
    model.record_data["AppliedF"].append(model.data.qfrc_applied.copy())


viz.run_simulation(muscle_force_pattern,
                   log_function=log_data_runtime,
                   record_function=record_data_runtime,
                   duration=5.0,
                   log_interval=1.0)


fig, axes = plt.subplots(4, 1, figsize=(10, 8))
# for i in range(sim.record_data["qpos"].shape[1]):
axes[0].plot(sim.record_data['time'], sim.record_data['qpos'][:, 7:9], linestyle='-', label=f'qpos[{[7,8]}]')
# axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
axes[0].grid(True)
axes[0].legend()
axes[1].plot(sim.record_data['time'], sim.record_data['AppliedF'], linestyle='-')
axes[1].grid(True)
axes[1].legend()
axes[2].plot(sim.record_data['time'], sim.record_data['mfrc'][:,27])
axes[2].grid(True)
axes[2].legend()

axes[3].plot(sim.record_data['time'], sim.record_data['ctrl'], label=f'ctrl[]')

plt.xlabel('Time [s]')
plt.ylabel('force')
plt.title('Joint Position over Time')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()




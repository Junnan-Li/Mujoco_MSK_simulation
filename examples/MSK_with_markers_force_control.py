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
# import src.utilities as ut


# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/hand/myohand_markers.xml')

viz = MusculoskeletalVisualizer(sim, azimuth=0, elevation=0,distance=1,lookat=[0.3, -0.25, 1.5])

# joint to be fixed 
jnt_lock_names = ['pro_sup','flexion','deviation']
jnt_lock_values = np.array([0,0,0])
sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)


# muscle force controller
muscle_names =  ['FDS2','FDP2','EDC2','EIP']
sim.control_act_index = sim.get_muscle_index(muscle_names)
# Define muscle activation pattern
def muscle_force_pattern(t:int, model:MusculoskeletalSimulation):
    """Generate coordinated muscle activation pattern"""
    n_muscles = model.n_actuators
    
    # Create wave-like activation pattern
    f_desired = np.zeros(n_muscles)
    # keep extension muscles stable forces
    f_desired[sim.control_act_index[1:4]] = 5

    f_MIF = model.get_muscle_MIF() 
    # Activate muscles in sequence
    wave_speed = 0.5  # Hz
    phase = 2 * np.pi *  wave_speed * t
    f_desired[sim.control_act_index[0]] = np.clip (8 +  5 * np.sin(phase), 0, f_MIF[sim.control_act_index[0]])
        
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
                "qvel": [],
                "ctrl": [],
                "act" : [],
                "mfrc": [],
                "mfrc_ss": [],
                "dfrc": []
            }
    model.record_data["time"].append(model.data.time)
    model.record_data["qpos"].append(model.data.qpos.copy())
    model.record_data["qvel"].append(model.data.qvel.copy())
    model.record_data["ctrl"].append(model.data.ctrl.copy())
    model.record_data["act"].append(model.data.act.copy())
    model.record_data["mfrc"].append(-model.data.actuator_force.copy())
    model.record_data["mfrc_ss"].append(-model.data.sensordata[4:8].copy())
    model.record_data["dfrc"].append(model.control_input.copy())


viz.run_simulation(muscle_force_pattern,
                   log_function=log_data_runtime,
                   record_function=record_data_runtime,
                   duration=10.0,
                   log_interval=1.0)


fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# for i in range(sim.record_data["qpos"].shape[1]):
axes[0].plot(sim.record_data['time'], sim.record_data['mfrc'][:, sim.control_act_index ], linestyle='--', label=f'mfrc[{sim.control_act_index }]')
axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
axes[0].grid(True)
axes[0].legend()
axes[1].plot(sim.record_data['time'], sim.record_data['mfrc'][:, sim.control_act_index[0] ], linestyle='--', label=f'mfrc[{sim.control_act_index }]')
axes[1].plot(sim.record_data['time'], sim.record_data['mfrc_ss'][:,0], label=f'dfrc[{sim.control_act_index }]')

plt.xlabel('Time [s]')
plt.ylabel('force')
plt.title('Joint Position over Time')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()




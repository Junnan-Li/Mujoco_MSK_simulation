import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer
import matplotlib.pyplot as plt

# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/muscle_test.xml')

# Set muscle control mode
muscle_params = {
    'activation_dynamics': False,
    'tau_activation': 0.015,
    'tau_deactivation': 0.060
}
sim.set_control_mode(ControlMode.MUSCLE, muscle_params)

# Initialize visualizer
viz = MusculoskeletalVisualizer(sim, azimuth=40,elevation=-20,distance=1)



# # change muscle activation dynamic parameters to converge faster
# sim.set_actuator_dynprm(0.005,0.01)
# print(sim.model.actuator_dynprm[0,0:2])

# Define muscle activation pattern
def muscle_activation_pattern(t:int, model:MusculoskeletalSimulation):
    """Generate coordinated muscle activation pattern"""
    n_muscles = model.n_actuators
    
    # Create wave-like activation pattern
    excitations = np.zeros(n_muscles)
    
    # Activate muscles in sequence
    wave_speed = 0.5  # Hz
    
    # phase = 2 * np.pi * (i / model.control_act_index.shape[0] + wave_speed * t)
    # excitations[i] = np.clip(0.3 + 0.5 * np.sin(phase), 0.0, 1.0)
    # excitations[i] = np.clip(0.3 + 0.5 * np.random.rand(), 0.0, 1.0)
    excitations += 0
        
    return np.clip(excitations, 0.0, 1.0)


def log_data_runtime(model:MusculoskeletalSimulation):
    activations = model.get_muscle_activations()
    forces = model.get_muscle_forces()
    site_dis = [np.linalg.norm(model.data.site(i).xpos - model.data.site(i+3).xpos)
                for i in range(3)]
    
    # Print activation summary
    print(f"Time: {model.data.time:.3f}")
    print(f"muscle activation: {activations}")
    print(f"muscle force: {forces}")
    print(f"site distance: {site_dis}")
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
                "site_dis":[]
            }

    site_dis = np.linalg.norm(model.data.site(0).xpos - model.data.site(1).xpos)
    model.record_data["time"].append(model.data.time)
    model.record_data["qpos"].append(model.data.qpos.copy())
    model.record_data["qvel"].append(model.data.qvel.copy())
    model.record_data["ctrl"].append(model.data.ctrl.copy())
    model.record_data["act"].append(model.data.act.copy())
    model.record_data["mfrc"].append(model.data.actuator_force.copy())
    model.record_data["site_dis"].append(site_dis)


# Run simulation
# print(f"Simulating {sim.n_muscles} muscles, {sim.n_joints} joints")
# print(f"Muscle names: {sim.muscle_names[:5]}...")  # Show first 5

viz.run_simulation(muscle_activation_pattern, log_data_runtime, record_data_runtime, duration=5.0, log_interval=1.0)

# print(f'{sim.record_data['time'].shape[0]}')

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(sim.record_data["ctrl"].shape[1]):
    # plt.plot(sim.record_data['time'], sim.record_data['ctrl'][:, i], label=f'ctrl[{i}]')
    # plt.plot(sim.record_data['time'], sim.record_data['act'][:, i], label=f'act[{i}]')
    axes[0].plot(sim.record_data['time'], sim.record_data['act'][:, i], label=f'act[{i}]')
axes[0].grid(True)
axes[0].legend()

for i in range(sim.record_data["site_dis"].shape[1]):
    axes[1].plot(sim.record_data['site_dis'], sim.record_data['mfrc'][:,i], label=f'mfrc')

plt.xlabel('site distance')
plt.ylabel('mscule force')
plt.title('muscle force vs. site distance')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()


# viz.close()
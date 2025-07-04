import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer
import matplotlib.pyplot as plt

# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/hand/myohand.xml')

# Set muscle control mode
muscle_params = {
    'activation_dynamics': True,
    'tau_activation': 0.015,
    'tau_deactivation': 0.060
}
sim.set_control_mode(ControlMode.MUSCLE, muscle_params)

# Initialize visualizer
viz = MusculoskeletalVisualizer(sim)


muscle_names =  ['FDS2','FDP2','EDC2','EIP']
sim.control_act_index = sim.get_muscle_index(muscle_names)

joint_fixed = []

# Define muscle activation pattern
def muscle_activation_pattern(t:int, model:MusculoskeletalSimulation):
    """Generate coordinated muscle activation pattern"""
    n_muscles = model.n_actuators
    
    # Create wave-like activation pattern
    excitations = np.zeros(n_muscles)
    
    # Activate muscles in sequence
    wave_speed = 0.5  # Hz
    if len(model.control_act_index)>0:
        for i in model.control_act_index:
            phase = 2 * np.pi * (i / model.control_act_index.shape[0] + wave_speed * t)
            excitations[i] = np.clip(0.3 + 0.5 * np.sin(phase), 0.0, 1.0)
        
    return np.clip(excitations, 0.0, 1.0)





# Run simulation
print(f"Simulating {sim.n_muscles} muscles, {sim.n_joints} joints")
# print(f"Muscle names: {sim.muscle_names[:5]}...")  # Show first 5

viz.run_simulation(muscle_activation_pattern, duration=5.0, log_interval=1.0)

# print(f'{sim.record_data['time'].shape[0]}')
# plot

plt.figure(figsize=(10, 5))
for i in range(sim.record_data["qpos"].shape[1]):
    plt.plot(sim.record_data['time'], sim.record_data['ctrl'][:, i], label=f'ctrl[{i}]')
    plt.plot(sim.record_data['time'], sim.record_data['mfrc'][:, i], label=f'mfrc[{i}]')
plt.xlabel('Time [s]')
plt.ylabel('Joint Position')
plt.title('Joint Position over Time')
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()


# viz.close()
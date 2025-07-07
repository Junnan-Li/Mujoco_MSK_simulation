import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer
import matplotlib.pyplot as plt
import mujoco

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

for _ in range(50):
    mujoco.mj_step(sim.model, sim.data)


MSK_state = sim.get_musculoskeletal_state()

muscle_activations = sim.get_muscle_activations()
muscle_forces = sim.get_muscle_forces()
qpos = sim.get_joint_positions()
qvel = sim.get_joint_velocities()
qtorque = sim.get_joint_torques()



# plot model information
print(f"{sim.n_joints} Joints: ")  
for i in range(sim.n_joints):
    print(f"Joint {i} name: {sim.joint_names[i]}           pose: {qpos[i]}    qtorque: {qtorque[i]}")  

print(f"{sim.n_muscles} Muscles: ")  
for i in range(sim.n_muscles):
    print(f"Muscle {i} name: {sim.muscle_names[i]}")  


# print(f"{sim.n_actuators} Actuators: ")  
# for i in range(sim.n_actuators):
#     print(f"Actuator {i} name: {sim.actuator_names[i]}")  















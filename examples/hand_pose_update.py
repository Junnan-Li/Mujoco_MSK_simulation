import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer

def sine_wave_joint_values(t):
    # Amplitudes and frequencies for each joint
    offsets = np.array([0.5,0,0.4,0.2])
    amplitudes = np.array([0.5, 0.1, 0.4, 0.2])
    frequencies = np.array([0.5, 0.3, 1, 0.25])  # in Hz
    phases = np.array([0, np.pi/2, np.pi, np.pi/4])

    # Compute joint values
    values = amplitudes * np.sin(2 * np.pi * frequencies * t + phases) + offsets
    return values

# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/hand/myohand.xml')

# Initialize visualizer
viz = MusculoskeletalVisualizer(sim)
viz.launch_viewer()
# Run simulation
print(f"Run Hand Pose Update script")

# choose joints to control 
joint_names =  ['mcp2_flexion','mcp2_abduction','pm2_flexion','md2_flexion']

sim_start_time = time.time()
sim_time = sim_start_time
duration = 5

while viz.is_running and (time.time() - sim_start_time) < duration:
    # print(f'{time.time()- sim_start_time}')
    t = time.time() - sim_start_time
    q_values = sine_wave_joint_values(t)

    sim.set_q_with_name(joint_names,q_values)
    viz.render()

    # Real-time synchronization
    if viz.real_time:
        elapsed = time.time() - sim_time
        sim_time = viz.sim.data.time - sim_start_time
        if sim_time > elapsed:
            time.sleep(sim_time - elapsed)
print('simulation finish')
# viz.close()
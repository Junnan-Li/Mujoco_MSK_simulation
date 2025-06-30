import numpy as np
from src.MSK_Model import MusculoskeletalSimulation, ControlMode
from src.visualizer import MusculoskeletalVisualizer


# Initialize simulation with MyoSuite-style model
sim = MusculoskeletalSimulation('./models/myo_sim/arm/myoarm.xml')

# Set muscle control mode
muscle_params = {
    'activation_dynamics': True,
    'tau_activation': 0.015,
    'tau_deactivation': 0.060
}
sim.set_control_mode(ControlMode.MUSCLE, muscle_params)

# Initialize visualizer
viz = MusculoskeletalVisualizer(sim)

# Define muscle activation pattern
def muscle_activation_pattern(t):
    """Generate coordinated muscle activation pattern"""
    n_muscles = sim.n_actuators
    
    # Create wave-like activation pattern
    excitations = np.zeros(n_muscles)
    
    # Activate muscles in sequence
    wave_speed = 2.0  # Hz
    for i in range(n_muscles):
        phase = 2 * np.pi * (i / n_muscles + wave_speed * t)
        excitations[i] = 0.3 + 0.3 * np.sin(phase)
        
    return np.clip(excitations, 0.0, 1.0)

# Run simulation
print(f"Simulating {sim.n_muscles} muscles, {sim.n_joints} joints")
print(f"Muscle names: {sim.muscle_names[:5]}...")  # Show first 5

viz.run_simulation(muscle_activation_pattern, duration=10.0, log_interval=2.0)
viz.close()
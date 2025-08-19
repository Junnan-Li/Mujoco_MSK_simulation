# Mujoco_MSK_simulation

A Python toolbox based on MuJoCo for musculoskeletal model simulation. 


## Initialize
Create a MusculoskeletalSimulation model

`sim = MusculoskeletalSimulation('./models/myo_sim/arm/myoarm.xml')`

Create a MusculoskeletalVisualizer object for the sim model with a defined view

`viz = MusculoskeletalVisualizer(sim, azimuth=90, elevation=0, distance=1,lookat=[0, -0.5, 1.2])`

## Run a forward simulation
define a controller, for example, a sin-wave muscle excitation function `muscle_activation_pattern(t, sim)`

[optional] define a log callback function `log_data_runtime(sim)`

[optional] define a record callback function `record_data_runtime(sim)`

Then call the simulation loop:

`viz.run_simulation(muscle_activation_pattern, log_data_runtime, record_data_runtime, duration=5.0, log_interval=1.0)`


### IK_Solver




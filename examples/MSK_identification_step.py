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

# joint to be fixed MCP

jnt_lock_names = ['pro_sup','flexion','deviation','pm2_flexion','md2_flexion'] #,'mcp2_abduction']
jnt_lock_values = np.array([np.pi/2,0,0,0,0])

# MCP PIP
# jnt_lock_names = ['pro_sup','flexion','deviation','md2_flexion'] #,'mcp2_abduction']
# jnt_lock_values = np.array([np.pi/2,0,0,0])

sim.lock_q_with_name(jnt_lock_names,jnt_lock_values)

# passive joint torque
jnt_passive = ['mcp2_flexion','pm2_flexion'] #,'pm2_flexion','md2_flexion']
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

    # time settings
    initial_time = 3 

    # # Activate muscles in sequence
    init_force = 4.0      # baseline force (N)
    step_interval = 2 # seconds per increment
    step_size = 1     # N per increment
    steps_increase = 7
    max_force = init_force + steps_increase * step_size   # N maximum

    # Create wave-like activation pattern
    f_desired = np.zeros(n_muscles)
    
    f_desired[sim.control_act_index[0:4]] = init_force
    f_desired[sim.control_act_index[3]] = 15

    if t > initial_time:    
        t_start = (t-initial_time)
        # Compute current force level
        step_index = int(t_start // step_interval)
        # Reset when exceeding max_force
        if step_index < steps_increase:
            f_desired[sim.control_act_index[0]] = init_force + step_size * step_index
        else:
            f_desired[sim.control_act_index[0]] = max_force - step_size * (step_index-steps_increase)


    return np.clip(f_desired, 0, f_MIF)




def log_data_runtime(model:MusculoskeletalSimulation):
    activations = model.get_muscle_activations()
    forces = model.get_muscle_forces()
    
    # Print activation summary
    print(f"Time: {model.data.time:.3f}")
    print(f"FDS muscle: {model.control_input[model.control_act_index[:]]} N")
    # print(f"muscle force: {forces}")
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
                   duration=40.0,
                   log_interval=1.0)

    
fig, axes = plt.subplots(figsize=(8, 6))
for i in range(len(muscle_names)):
    axes[0].scatter(sim.record_data['qpos'][:, jnt_passive_index[0]], sim.record_data['tendon_pos'][:,i], s=5, label=f'tendon length {muscle_names[i]}')
axes[0].set_ylabel("joint pos [rad]")
axes[0].grid(True)
axes[0].legend()


# tendon length functions
tendon_length_funcs = []     # store each dpoly (function)
tendon_length_coeffs = []    # store coefficients for saving if needed
tendon_length_values = []    # store evaluated values dpoly(q)
# moment arm estimation  
moment_arm_funcs = []     # store each dpoly (function)
moment_arm_coeffs = []    # store coefficients for saving if needed
moment_arm_values = []    # store evaluated values dpoly(q)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(len(muscle_names)):
    q = sim.record_data['qpos'][:, jnt_passive_index[0]]      # joint angle
    l_t = sim.record_data['tendon_pos'][:, i]   
    
    # Fit tendon length vs. joint angle (cubic)
    coeffs = np.polyfit(q, l_t, 3)
    poly = np.poly1d(coeffs)

    tendon_length_funcs.append(poly)
    tendon_length_coeffs.append(poly.coeffs)

    dpoly = np.polyder(poly)   # derivative function (moment arm)
    
    # Store everything
    moment_arm_funcs.append(dpoly)
    moment_arm_coeffs.append(dpoly.coeffs)
    moment_arm_values.append(dpoly(q)) 

    axes[0].plot(np.sort(q) , -dpoly(np.sort(q)),label=f' {muscle_names[i]}')
axes[0].grid(True)
axes[0].set_ylabel('moment arm calculated' )
axes[0].legend()

for i in range(len(muscle_names)):
    momentarm = sim.get_muscle_moment_arms_curves(['mcp2_flexion'],[muscle_names[i]] )
    q_deg = momentarm[:, 0]
    r = momentarm[:, 1]
    axes[1].plot(q_deg , r,label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel('moment arm true' )
axes[1].legend()

plt.xlabel('q [rad]')
plt.grid(True)
# plt.tight_layout()
# plt.show()

# %%
#  q(t) and f(t)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(jnt_passive_index.shape[0]):
    axes[0].plot(sim.record_data['time'], sim.record_data['qpos'][:, jnt_passive_index[i]], linestyle='-', label=f'q {jnt_passive[i]}')
# axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
# axes[0].set_title("joint pos")
axes[0].set_ylabel("joint pos [rad]")
axes[0].grid(True)
axes[0].legend()

for i in range(len(muscle_names)):
    axes[1].plot(sim.record_data['time'], sim.record_data['mfrc'][:,sim.control_act_index[i]],label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel(' tendon force [N] ' )
axes[1].legend()
plt.xlabel('time [s]')
# plt.ylabel('force')
# plt.legend()
plt.grid(True)
plt.suptitle('stepwise ID: q(t) and f(t)')

# fit passive stiffness

time = sim.record_data['time']
q = sim.record_data['qpos'][:, jnt_passive_index[0]]              # (T,)
f_all = sim.record_data['mfrc'][:, sim.control_act_index]         # (T, M)
n_muscles = len(muscle_names)

# Prepare arrays
moment_arms = np.zeros_like(f_all)   # (T, M)

# --- compute moment arm N_i(q) for each tendon i ---
for i in range(n_muscles):
    l_t = sim.record_data['tendon_pos'][:, i]
    coeffs = np.polyfit(q, l_t, 3)     # cubic fit
    dpoly = np.polyder(np.poly1d(coeffs))
    moment_arms[:, i] = dpoly(q)       # evaluate at each recorded q

# --- compute total passive joint torque ---
# h(t) = sum_i N_i(q_t) * f_i(t)
h_t  = np.sum(moment_arms * f_all, axis=1)   # (T,)
# to function of q
order = np.argsort(q)
q_sorted  = q[order]
h_sorted  = h_t[order]

# Fit cubic: h(q) ≈ a3 q^3 + a2 q^2 + a1 q + a0
h_coeffs = np.polyfit(q_sorted, h_sorted, 3)
h_poly   = np.poly1d(h_coeffs)   # callable: h_poly(q_query)

plt.figure(figsize=(8,5))
plt.plot(q, h_poly(q),'g-', label='estimated h(q)')
plt.plot(q, ut.tau_stiff_DE(q, A, B, C, D, E, F), 'b-' , label='true h(q)')
plt.ylabel(' tendon force [N] ' )
plt.legend()
plt.grid(True)
plt.xlabel('q [rad]')
plt.title('estimated h(q) and ground truth')
    

## save data
q        = sim.record_data['qpos'][:, jnt_passive_index[0]]
t        = sim.record_data['time']
f_all    = sim.record_data['mfrc'][:, sim.control_act_index]
record   = sim.record_data

# Save multiple arrays
np.savez('ID_step_MCP.npz',
         time=time,
         q=q,
         f_all=f_all,
         record = record,
         muscle_names=muscle_names,
         tendon_length_coeffs=tendon_length_coeffs,
         moment_arm_coeffs=moment_arm_coeffs,
         h_MCP_coeffs=h_coeffs)



# %%

data = np.load('ID_step_MCP.npz', allow_pickle=True)
q = data['q']
time = data['time']
f_all = data['f_all']
muscle_names = data['muscle_names']
tendon_length_funcs = [np.poly1d(c) for c in data['tendon_length_coeffs']]
moment_arm_funcs = [np.poly1d(c) for c in data['moment_arm_coeffs']]
h_poly = np.poly1d(data['h_MCP_coeffs'])

# Example: evaluate at new q values
q_test = np.linspace(q.min(), q.max(), 100)
h_est = h_poly(q_test)

# get Moment arm matrix of PIP joint
# tendon length functions
tendon_length_funcs_PIP = []     # store each dpoly (function)
tendon_length_coeffs_PIP = []    # store coefficients for saving if needed
# moment arm estimation  
moment_arm_funcs_PIP = []     # store each dpoly (function)
moment_arm_coeffs_PIP = []    # store coefficients for saving if needed

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(len(muscle_names)):
    q = sim.record_data['qpos'][:, jnt_passive_index[1]]      # joint angle
    q_mcp = sim.record_data['qpos'][:, jnt_passive_index[0]]  
    l_t = sim.record_data['tendon_pos'][:, i] - np.poly1d(tendon_length_funcs[i])(q_mcp)
    
    # Fit tendon length vs. joint angle (cubic)
    coeffs = np.polyfit(q, l_t, 3)
    poly = np.poly1d(coeffs)

    tendon_length_funcs_PIP.append(poly)
    tendon_length_coeffs_PIP.append(poly.coeffs)

    dpoly = np.polyder(poly)   # derivative function (moment arm)
    
    # Store everything
    moment_arm_funcs_PIP.append(dpoly)
    moment_arm_coeffs_PIP.append(dpoly.coeffs)

    axes[0].plot(np.sort(q) , -dpoly(np.sort(q)),label=f' {muscle_names[i]}')
axes[0].grid(True)
axes[0].set_ylabel('moment arm calculated' )
axes[0].legend()

for i in range(len(muscle_names)):
    momentarm = sim.get_muscle_moment_arms_curves([jnt_passive[1]],[muscle_names[i]] )
    q_deg = momentarm[:, 0]
    r = momentarm[:, 1]
    axes[1].plot(q_deg , r,label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel('moment arm true' )
axes[1].legend()

plt.xlabel('q [rad]')
plt.grid(True)

# q(t) and f(t)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
for i in range(jnt_passive_index.shape[0]):
    axes[0].plot(sim.record_data['time'], sim.record_data['qpos'][:, jnt_passive_index[i]], linestyle='-', label=f'q {jnt_passive[i]}')
# axes[0].plot(sim.record_data['time'], sim.record_data['dfrc'][:, sim.control_act_index ], label=f'dfrc[{sim.control_act_index }]')
# axes[0].set_title("joint pos")
axes[0].set_ylabel("joint pos [rad]")
axes[0].grid(True)
axes[0].legend()

for i in range(len(muscle_names)):
    axes[1].plot(sim.record_data['time'], sim.record_data['mfrc'][:,sim.control_act_index[i]],label=f' {muscle_names[i]}')
axes[1].grid(True)
axes[1].set_ylabel(' tendon force [N] ' )
axes[1].legend()
plt.xlabel('time [s]')
# plt.ylabel('force')
# plt.legend()
plt.grid(True)
plt.suptitle('stepwise ID: q(t) and f(t)')


# %% 
# fit passive stiffness

q = sim.record_data['qpos'][:, jnt_passive_index[1]]              # (T,)
f_all = sim.record_data['mfrc'][:, sim.control_act_index]         # (T, M)
n_muscles = len(muscle_names)

# Prepare arrays
moment_arms = np.zeros_like(f_all)   # (T, M)

# --- compute moment arm N_i(q) for each tendon i ---
for i in range(n_muscles):
    dpoly = moment_arm_funcs_PIP[i]
    moment_arms[:, i] = dpoly(q)       # evaluate at each recorded q

# --- compute total passive joint torque ---
# h(t) = sum_i N_i(q_t) * f_i(t)
h_t_PIP  = np.sum(moment_arms * f_all, axis=1)   # (T,)
# to function of q
order = np.argsort(q)
q_sorted  = q[order]
h_sorted  = h_t_PIP[order]

# Fit cubic: h(q) ≈ a3 q^3 + a2 q^2 + a1 q + a0
h_coeffs_PIP = np.polyfit(q_sorted, h_sorted, 3)
h_poly_PIP   = np.poly1d(h_coeffs_PIP)   # callable: h_poly(q_query)

plt.figure(figsize=(8,5))
plt.plot(q, h_poly_PIP(q),'g-', label='estimated h(q)')
plt.plot(q, ut.tau_stiff_DE(q, A, B, C, D, E, F), 'b-' , label='true h(q)')
plt.ylabel(' tendon force [N] ' )
plt.legend()
plt.grid(True)
plt.xlabel('q [rad]')
plt.title('estimated h(q) and ground truth')
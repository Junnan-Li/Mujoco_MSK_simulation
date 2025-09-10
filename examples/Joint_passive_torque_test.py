import numpy as np
import matplotlib.pyplot as plt
import src.utilities as ut

# Parameters
A = 1.27
B = 0.031
C = 1.83
D = 0.07
E = 47.39   # neutral for first exponential
F = 58.97   # neutral for second exponential

# Joint angle range (radians)
q_vals = np.linspace(-np.pi/2, np.pi/2, 1000)

# Compute torques
tau_vals = ut.tau_stiff_DE(q_vals, A, B, C, D, E, F)

# Plot
plt.figure(figsize=(6,4))
plt.plot(q_vals, tau_vals, label="Double exponential torque", linestyle="--")
plt.axhline(0, color="k", linewidth=0.5)
plt.axvline(0, color="k", linewidth=0.5)
plt.xlabel("Joint angle q (rad)")
plt.ylabel("Torque τ (Nm)")
plt.title("Passive Joint Torque: Double Exponential")
plt.legend()
plt.grid(True)
plt.show()
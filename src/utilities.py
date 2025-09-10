import mujoco
import mujoco.viewer
import numpy as np
# from typing import List
from scipy.spatial.transform import Rotation as R

def draw_frame(viewer: mujoco.viewer, 
               origin:np.ndarray, 
               rotv:np.ndarray,  # rotation vector
               AxisLen: float=0.1):
        """Draw RGB axis lines at given position and orientation."""
        pass
        colors = [(1, 0, 0, 1),  # X - red
                (0, 1, 0, 1),  # Y - green
                (0, 0, 1, 1)]  # Z - blue
        rotm = R.from_rotvec(rotv).as_matrix()        
        for i in range(3):
            # add one user_scn.geom
            viewer.user_scn.ngeom += 1
            arrow_to = origin + AxisLen * rotm[:, i]
            mujoco.mjv_initGeom(
                geom=viewer.user_scn.geoms[viewer.user_scn.ngeom-1],
                type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.zeros(9),
                rgba=np.array(colors[i])
            )
            mujoco.mjv_connector(
                geom=viewer.user_scn.geoms[viewer.user_scn.ngeom-1],
                type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                width=0.02*AxisLen,
                from_=origin,
                to=arrow_to)
            

# double exponential joint stiffness torque 
def tau_stiff_DE(q, A, B, C, D, E, F):
    """
    Double-Exponential passive joint torque
    Input: 
        q: joint angle in rad
        A-F: parameters
    Output:
        tau: in Nm
    """
    Tau_stiff = (A * (np.exp(-B * (180*q/np.pi - E)) - 1.0) - C * (np.exp(D * (180*q/np.pi - F)) - 1.0)) /100
    return Tau_stiff

# joint damping
def tau_damping(q_vel, k ):
    """
    linear joint damping torque
    Input: 
        q_vel: joint angle in rad/s
        k: parameters
    Output:
        tau: in N/m/s
    """
    return -q_vel * k
import mujoco
import numpy as np
from enum import Enum
from typing import Optional, Dict, Any, List
from src.OptimParams import OptimParams


class ControlMode(Enum):
    MUSCLE = "muscle"          # Direct muscle activation
    JOINT = "joint"           # Joint-level control
    EE_Pos = "EE_position"         # end effector position

class MusculoskeletalSimulation:
    """Main musculoskeletal simulation class compatible with MyoSuite models"""
    
    def __init__(self, xml_path: str):
        """
        Initialize musculoskeletal simulation
        
        Args:
            xml_path: Path to MuJoCo XML model file (MyoSuite format)
        """
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Parse musculoskeletal structure
        self._parse_musculoskeletal_structure()
        
        # do forward simulation
        self.integrate = True

        # if joints are limited 
        self.jnt_lock = False
        self.jnt_lock_id = []
        self.jnt_lock_value = []

        # Control mode and controller
        self.control_mode = None
        self.controller = None
        self.control_act_index = []

        # Store initial state
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        # Store simulation state
        # need define callback function
        self.record_data = {}
            



    def _parse_musculoskeletal_structure(self):
        """Parse muscle and joint information from the model"""
        # Get muscle information (tendons in MuJoCo represent muscles)
        self.n_muscles = self.model.ntendon
        self.muscle_names = []
        for i in range(self.n_muscles):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_TENDON, i)
            self.muscle_names.append(name if name else f"muscle_{i}")
            
        # Get joint information
        self.n_joints = self.model.nq
        self.joint_names = []
        for i in range(self.n_joints):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_names.append(name if name else f"joint_{i}")
            
        # Get actuator information (muscle actuators)
        self.n_actuators = self.model.nu
        self.actuator_names = []
        for i in range(self.n_actuators):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.actuator_names.append(name if name else f"actuator_{i}")

        # # Get site information (muscle actuators)
        # self.n_actuators = self.model.nu
        # self.actuator_names = []
        # for i in range(self.n_actuators):
        #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        #     self.actuator_names.append(name if name else f"actuator_{i}")
            
    def set_control_mode(self, mode: ControlMode, controller_params: Optional[Dict] = None):
        """
        Set control mode and initialize corresponding controller
        
        Args:
            mode: Control mode (MUSCLE, JOINT, or REFLEX)
            controller_params: Parameters for the controller
        """
        self.control_mode = mode
        
        if mode == ControlMode.MUSCLE:
            from src.controllers.muscle_controller import MuscleController
            self.controller = MuscleController(self.model, self.data, controller_params)
            
        elif mode == ControlMode.JOINT:
            from src.controllers.joint_controller import JointController
            self.controller = JointController(self.model, self.data, controller_params)
            
        # elif mode == ControlMode.REFLEX:
        #     from controllers.reflex_controller import ReflexController
        #     self.controller = ReflexController(self.model, self.data, controller_params)

    def set_q_with_name(self, q_names, q_values):
        # update the q with given names
        name_to_index = {}
        for name in q_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f"Joint '{name}' not found.")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qpos_dim = 1 if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE else 7
            if qpos_dim > 1:
                raise ValueError(f"Joint '{name}' has more than 1 DoF.")
            name_to_index[name] = qpos_addr

        for name, val in zip(q_names, q_values):
            index = name_to_index[name]
            self.data.qpos[index] = val
            
        mujoco.mj_forward(self.model, self.data)

    def lock_q_with_name(self, q_names: List[str], q_values: np.ndarray):

        for i in range(len(q_names)):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, q_names[i])
            if joint_id == -1:
                raise ValueError(f"Joint '{q_names[i]}' not found.")
            self.jnt_lock_id.append(self.model.jnt_qposadr[joint_id])
            self.jnt_lock_value.append(q_values[i])

        self.jnt_lock = True
          
    ## actuator settings
    def set_actuator_dynprm(self, tau_up, tau_down):
        """
        set tau_up and tau_down of muscle activation dynamics
        
        usage: 
            sim.set_actuator_dynprm(0.005,0.01)
        """
        # set muscle activation dynamic parameters
        
        for i in range(self.model.nu):
            if self.model.actuator_trntype[i] == 3:
                # actuator is muscle type
                self.model.actuator_dynprm[i,0:2] = np.array([tau_up,tau_down])


    def step(self, control_input: np.ndarray):
        """
        Execute one simulation step with control input
        
        Args:
            control_input: Control command based on current control mode
        """
        if self.controller is None:
            raise RuntimeError("Control mode not set. Call set_control_mode() first.")
            
        # Apply control through the controller
        self.controller.apply_control(control_input)
        
        if self.jnt_lock:
            # set qpos to desired value
            for i in range(len(self.jnt_lock_id)):
                self.data.qpos[self.jnt_lock_id[i]] = self.jnt_lock_value[i]

        if self.integrate:
            #  Step the simulation
            mujoco.mj_step(self.model, self.data)
        else:
            # only forward without integrate
            mujoco.mj_forward(self.model, self.data)
        

        
    def reset(self):
        """Reset simulation to initial state"""
        # 
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)
        
        if self.controller:
            self.controller.reset()
            


    # 
    def get_site_Jac(self, site_name: str) -> np.ndarray:
        jac = np.empty((6,self.model.nv))
        # jacr = np.empty(3,len(site_name))
        # for i in range(len(site_name)):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,site_name)
        if  site_id == -1:
            raise ValueError(f"Site '{site_name}' not found.")
        
        mujoco.mj_jacSite(self.model, self.data, jac[:3], jac[3:], site_id) 
        return jac

    def ik_site(self, site_name: str="", method: str="NR", optprm: OptimParams=OptimParams()):
        pass

    def get_muscle_index(self, muscle_names: List[str]) -> np.ndarray:
        indices = []
        for name in muscle_names:
            try:
                idx = self.model.actuator(name).id  
                indices.append(idx)
            except KeyError:
                print(f"Muscle with name '{name}' does not exist")
        return np.array(indices, dtype=int)

    # Musculoskeletal-specific getters
    def get_muscle_activations(self) -> np.ndarray:
        """Get current muscle activations"""
        return self.data.ctrl.copy()
        
    def get_muscle_forces(self) -> np.ndarray:
        """Get current muscle forces"""
        return self.data.actuator_force.copy()
        
    def get_muscle_lengths(self) -> np.ndarray:
        """Get current muscle lengths"""
        return self.data.ten_length.copy()
        
    def get_muscle_velocities(self) -> np.ndarray:
        """Get current muscle velocities"""
        return self.data.ten_velocity.copy()
        
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return self.data.qpos.copy()
        
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        return self.data.qvel.copy()
        
    def get_joint_torques(self) -> np.ndarray:
        """Get current joint torques"""
        return self.data.qfrc_actuator.copy()
        
    def get_end_effector_pos(self, body_name: str) -> np.ndarray:
        """Get end effector position"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()
        
    def get_musculoskeletal_state(self) -> Dict[str, Any]:
        """Get complete musculoskeletal state"""
        return {
            'time': self.data.time,
            'joint_pos': self.get_joint_positions(),
            'joint_vel': self.get_joint_velocities(),
            'joint_torques': self.get_joint_torques(),
            'muscle_activations': self.get_muscle_activations(),
            'muscle_forces': self.get_muscle_forces(),
            'muscle_lengths': self.get_muscle_lengths(),
            'muscle_velocities': self.get_muscle_velocities()
        }
        
    def get_muscle_moment_arms(self, joint_names: List[str]) -> np.ndarray:
        """
        Get muscle moment arms for specified joints
        
        Args:
            joint_names: List of joint names
            
        Returns:
            Matrix of moment arms [n_muscles x n_joints]
        """
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                    for name in joint_names]
        
        moment_arms = np.zeros((self.n_muscles, len(joint_ids)))
        
        for i, joint_id in enumerate(joint_ids):
            # Calculate moment arms using MuJoCo's jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            
            for muscle_id in range(self.n_muscles):
                # Get tendon wrapping points and compute moment arm
                # This is a simplified version - full implementation would use
                # MuJoCo's tendon jacobian functionality
                moment_arms[muscle_id, i] = self.data.ten_wrapadr[muscle_id] if muscle_id < len(self.data.ten_wrapadr) else 0.0
                
        return moment_arms
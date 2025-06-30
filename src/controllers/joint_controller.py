import mujoco
import numpy as np
from typing import Dict, Optional

class JointController:
    """Joint-level controller that converts joint commands to muscle activations"""
    
    def __init__(self, model, data, params: Optional[Dict] = None):
        self.model = model
        self.data = data
        
        if params is None:
            params = {}
            
        # PID gains for joint control
        self.kp = params.get('kp', 100.0)
        self.ki = params.get('ki', 1.0)
        self.kd = params.get('kd', 10.0)
        
        # Control allocation parameters
        self.use_torque_to_muscle = params.get('use_torque_to_muscle', True)
        self.antagonist_ratio = params.get('antagonist_ratio', 0.1)
        
        # Control state
        self.prev_error = np.zeros(model.nq)
        self.integral_error = np.zeros(model.nq)
        self.dt = model.opt.timestep
        
        # Muscle grouping (simplified: assume muscles come in agonist-antagonist pairs)
        self.muscle_groups = self._identify_muscle_groups()
        
    def _identify_muscle_groups(self):
        """Identify muscle groups based on moment arms"""
        # Simplified muscle grouping - in practice, this would use
        # moment arm analysis to group muscles by joint action
        n_joints = self.model.nq
        n_muscles = self.model.nu
        
        muscles_per_joint = n_muscles // n_joints if n_joints > 0 else 1
        
        groups = {}
        for joint_idx in range(min(n_joints, n_muscles)):
            start_idx = joint_idx * muscles_per_joint
            end_idx = min(start_idx + muscles_per_joint, n_muscles)
            groups[f'joint_{joint_idx}'] = list(range(start_idx, end_idx))
            
        return groups
        
    def apply_control(self, target_positions: np.ndarray):
        """
        Apply joint position control converted to muscle activations
        
        Args:
            target_positions: Desired joint positions
        """
        n_controllable_joints = min(len(target_positions), self.model.nq)
        current_positions = self.data.qpos[:n_controllable_joints]
        
        # PID control to get desired joint torques
        error = target_positions[:n_controllable_joints] - current_positions
        self.integral_error[:n_controllable_joints] += error * self.dt
        derivative_error = (error - self.prev_error[:n_controllable_joints]) / self.dt
        
        desired_torques = (self.kp * error + 
                          self.ki * self.integral_error[:n_controllable_joints] + 
                          self.kd * derivative_error)
        
        if self.use_torque_to_muscle:
            # Convert torques to muscle activations
            muscle_activations = self._torque_to_muscle_activations(desired_torques)
        else:
            # Simple mapping (for models with direct joint actuators)
            muscle_activations = np.clip(desired_torques / 100.0, 0.0, 1.0)
            
        # Apply muscle activations
        if len(muscle_activations) <= self.model.nu:
            self.data.ctrl[:len(muscle_activations)] = muscle_activations
        else:
            self.data.ctrl[:] = muscle_activations[:self.model.nu]
            
        # Update control history
        self.prev_error[:n_controllable_joints] = error
        
    def _torque_to_muscle_activations(self, desired_torques: np.ndarray) -> np.ndarray:
        """
        Convert desired joint torques to muscle activations
        
        Args:
            desired_torques: Desired torques for each joint
            
        Returns:
            Muscle activations
        """
        muscle_activations = np.zeros(self.model.nu)
        
        for joint_idx, torque in enumerate(desired_torques):
            joint_key = f'joint_{joint_idx}'
            if joint_key in self.muscle_groups:
                muscle_indices = self.muscle_groups[joint_key]
                
                if len(muscle_indices) >= 2:
                    # Agonist-antagonist muscle pair
                    if torque >= 0:
                        # Positive torque: activate agonist
                        muscle_activations[muscle_indices[0]] = min(abs(torque) / 50.0, 1.0)
                        muscle_activations[muscle_indices[1]] = self.antagonist_ratio
                    else:
                        # Negative torque: activate antagonist
                        muscle_activations[muscle_indices[0]] = self.antagonist_ratio
                        muscle_activations[muscle_indices[1]] = min(abs(torque) / 50.0, 1.0)
                else:
                    # Single muscle
                    muscle_activations[muscle_indices[0]] = min(abs(torque) / 50.0, 1.0)
                    
        return muscle_activations
        
    def reset(self):
        """Reset controller state"""
        self.prev_error.fill(0)
        self.integral_error.fill(0)
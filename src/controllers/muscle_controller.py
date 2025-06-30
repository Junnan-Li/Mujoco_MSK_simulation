import mujoco
import numpy as np
from typing import Dict, Optional

class MuscleController:
    """Direct muscle activation controller for musculoskeletal models"""
    
    def __init__(self, model, data, params: Optional[Dict] = None):
        self.model = model
        self.data = data
        
        if params is None:
            params = {}
            
        # Muscle activation parameters
        self.activation_dynamics = params.get('activation_dynamics', True)
        self.tau_act = params.get('tau_activation', 0.015)  # Activation time constant
        self.tau_deact = params.get('tau_deactivation', 0.060)  # Deactivation time constant
        
        # Muscle activation state
        self.muscle_activations = np.zeros(model.nu)
        self.muscle_excitations = np.zeros(model.nu)
        
        # Muscle force scaling
        self.max_force_scaling = params.get('max_force_scaling', 1.0)
        
    def apply_control(self, excitations: np.ndarray):
        """
        Apply muscle excitation control with activation dynamics
        
        Args:
            excitations: Muscle excitations (0-1)
        """
        if len(excitations) != self.model.nu:
            raise ValueError(f"Expected {self.model.nu} excitations, got {len(excitations)}")
            
        # Clip excitations to valid range
        excitations = np.clip(excitations, 0.0, 1.0)
        self.muscle_excitations = excitations
        
        if self.activation_dynamics:
            # Apply activation dynamics
            dt = self.model.opt.timestep
            
            for i in range(len(excitations)):
                if excitations[i] > self.muscle_activations[i]:
                    # Activation
                    tau = self.tau_act
                else:
                    # Deactivation
                    tau = self.tau_deact
                    
                # First-order dynamics
                self.muscle_activations[i] += (excitations[i] - self.muscle_activations[i]) * dt / tau
                
            # Clip activations
            self.muscle_activations = np.clip(self.muscle_activations, 0.0, 1.0)
        else:
            # Direct activation (no dynamics)
            self.muscle_activations = excitations
            
        # Apply muscle activations to MuJoCo control
        self.data.ctrl[:] = self.muscle_activations * self.max_force_scaling
        
    def get_muscle_activations(self) -> np.ndarray:
        """Get current muscle activations"""
        return self.muscle_activations.copy()
        
    def get_muscle_excitations(self) -> np.ndarray:
        """Get current muscle excitations"""
        return self.muscle_excitations.copy()
        
    def reset(self):
        """Reset controller state"""
        self.muscle_activations.fill(0)
        self.muscle_excitations.fill(0)
import mujoco
import mujoco.viewer
import numpy as np
from typing import Optional, Callable
import time

class MusculoskeletalVisualizer:
    """Enhanced visualizer for musculoskeletal simulations with muscle activation display"""
    
    def __init__(self, myoskel_sim):
        """
        Initialize musculoskeletal visualizer
        
        Args:
            myoskel_sim: MusculoskeletalSimulation instance
        """
        self.sim = myoskel_sim
        self.viewer = None
        self.is_running = False
        
        # Visualization options
        self.show_muscle_forces = True
        self.show_activation_colors = True
        self.force_scale = 0.001  # Scale factor for force visualization
        
    def launch_viewer(self):
        """Launch the MuJoCo viewer with muscle visualization"""
        self.viewer = mujoco.viewer.launch_passive(
            self.sim.model, 
            self.sim.data,
            show_left_ui=False,show_right_ui=False
        )
        self.is_running = True
        
        # Configure viewer for muscle visualization
        self._configure_muscle_visualization()
        
    def _configure_muscle_visualization(self):
        """Configure viewer for optimal muscle visualization"""
        if self.viewer:
            # Enable tendon visualization
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
            
            # Set transparency for better muscle visibility
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            
            # set focus
            self.viewer.cam.azimuth = 180      # Rotate camera 90 degrees around model
            self.viewer.cam.elevation = 0   # Tilt camera downward
            self.viewer.cam.distance = 1    # Zoom out
            self.viewer.cam.lookat[:] = [0.7, 0, 1]  # Center on torso
            
    def render(self):
        """Render current frame with muscle activation coloring"""
        if self.viewer and self.is_running:
            if self.show_activation_colors:
                self._update_muscle_colors()
            self.viewer.sync()
            
    def _update_muscle_colors(self):
        """Update muscle colors based on activation levels"""
        activations = self.sim.get_muscle_activations()
        
        # Color muscles based on activation (red = high activation, blue = low activation)
        for i in range(min(len(activations), self.sim.model.ntendon)):
            activation = activations[i] if i < len(activations) else 0.0
            
            # Set tendon color based on activation
            # This would require custom rendering - simplified version here
            pass
            
    def plot_muscle_activations(self):
        """Plot real-time muscle activations (would require matplotlib integration)"""
        activations = self.sim.get_muscle_activations()
        forces = self.sim.get_muscle_forces()
        
        # Print activation summary
        print(f"Time: {self.sim.data.time:.3f}")
        print(f"Max activation: {np.max(activations):.3f}")
        print(f"Mean activation: {np.mean(activations):.3f}")
        print(f"Active muscles: {np.sum(activations > 0.1)}/{len(activations)}")
        print("---")
        
    def run_simulation(self, 
                      control_function: Callable[[float], np.ndarray],
                      duration: float = 10.0,
                      real_time: bool = True,
                      log_interval: float = 1.0):
        """
        Run simulation with musculoskeletal visualization
        
        Args:
            control_function: Function that takes time and returns control input
            duration: Simulation duration in seconds
            real_time: Whether to run in real time
            log_interval: Interval for logging muscle state
        """
        if not self.is_running:
            self.launch_viewer()
            
            
        start_time = time.time()
        sim_start_time = self.sim.data.time
        last_log_time = 0
        
        while self.is_running and (self.sim.data.time - sim_start_time) < duration:
            # Get control input
            current_time = self.sim.data.time
            control_input = control_function(current_time)
            
            # Step simulation
            self.sim.step(control_input)
            
            # Render
            self.render()
            
            # Log muscle state
            if current_time - last_log_time >= log_interval:
                self.plot_muscle_activations()
                last_log_time = current_time
                
            # Real-time synchronization
            if real_time:
                elapsed = time.time() - start_time
                sim_time = self.sim.data.time - sim_start_time
                if sim_time > elapsed:
                    time.sleep(sim_time - elapsed)
                    
    def close(self):
        """Close the viewer"""
        if self.viewer:
            self.viewer.close()
            self.is_running = False
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
        self.opt = None
        self.is_running = False

        self.real_time = True
        
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

        # viewer opt settings
        self.viewer.opt.geomgroup[1] = False
        # self.viewer.opt.geomgroup[2] = False
        
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0    # Show convex hulls
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 0       # Show textures
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0         # Show joints
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1        # Hide actuators
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 0        # Hide cameras
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0         # Hide lights
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1        # Show tendons
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0   # Hide range finders
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1    # Show constraints
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0       # Hide inertia boxes
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 0    # Hide scaled inertia
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0     # Hide perturbation forces
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0       # Hide perturbation object
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # Show contact points
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # Show contact forces
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0  # Hide contact splits
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0   # Disable transparency
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = 1   # Auto-connect bodies
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0           # Hide center of mass
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 1        # Show selection
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1        # Show static bodies
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1          # Show skin

        # set focus
        self.viewer.cam.azimuth = 180      # Rotate camera 90 degrees around model
        self.viewer.cam.elevation = 0   # Tilt camera downward
        self.viewer.cam.distance = 0.1    # Zoom out
        self.viewer.cam.lookat[:] = [0.4, -0.25, 1.5]  # Center on torso

        

        self.is_running = True

        # Configure viewer for muscle visualization
        # self._configure_muscle_visualization()
        
    def _configure_muscle_visualization(self):
        """Not used. Configure viewer for optimal muscle visualization"""
        if self.viewer:
            # Enable tendon visualization
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
            
            # Set transparency for better muscle visibility
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            
            
    def render(self):
        """Render current frame with muscle activation coloring"""
        
        if self.show_activation_colors:
                self._update_muscle_colors()
        self.viewer.sync()
            
    def _update_muscle_colors(self):
        """Update muscle colors based on activation levels"""
        # get ctrl
        excitation = self.sim.get_muscle_activations().reshape(-1, 1)
        num_excitation = excitation.shape[0]
        if num_excitation == self.sim.model.ntendon:
            # Color muscles based on activation (red = high activation, blue = low activation)
            tendon_colors = np.hstack([excitation,      # R
                        np.zeros_like(excitation),      # G
                        1 - excitation,                 # B
                        np.ones_like(excitation)])      # A
            self.sim.model.tendon_rgba = tendon_colors
        else:
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

    def log_data_step(self):
        """
        Data record
        """
        self.sim.record_data["time"].append(self.sim.data.time)
        self.sim.record_data["qpos"].append(self.sim.data.qpos.copy())
        self.sim.record_data["qvel"].append(self.sim.data.qvel.copy())
        self.sim.record_data["ctrl"].append(self.sim.data.ctrl.copy())
        self.sim.record_data["mfrc"].append(self.sim.data.actuator_force.copy())

    def transfer_data_to_np(self):
        """
        Data record
        """
        for key in self.sim.record_data:
            self.sim.record_data[key] = np.array(self.sim.record_data[key])


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
        with mujoco.viewer.launch_passive(
            self.sim.model, 
            self.sim.data,
            show_left_ui=False,show_right_ui=False
        ) as self.viewer:

            # viewer opt settings
            # self.viewer.opt.geomgroup[1] = False
            # self.viewer.opt.geomgroup[2] = False
            
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0    # Show convex hulls
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 0       # Show textures
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0         # Show joints
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1        # Hide actuators
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1        # Hide cameras
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0         # Hide lights
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1        # Show tendons
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0   # Hide range finders
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1    # Show constraints
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0       # Hide inertia boxes
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 0    # Hide scaled inertia
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0     # Hide perturbation forces
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0       # Hide perturbation object
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # Show contact points
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # Show contact forces
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0  # Hide contact splits
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0   # Disable transparency
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = 1   # Auto-connect bodies
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0           # Hide center of mass
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 1        # Show selection
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1        # Show static bodies
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1          # Show skin

            # set focus
            self.viewer.cam.azimuth = 180      # 
            self.viewer.cam.elevation = 0   # Tilt camera downward
            self.viewer.cam.distance = .1    # Zoom out
            self.viewer.cam.lookat[:] = [0.4, -0.25, 1.5]  # Center on torso

            # self.is_running = True

        
            start_time = time.time()
            sim_start_time = self.sim.data.time
            last_log_time = 0
            
            while (self.sim.data.time - sim_start_time) < duration:
                # Get control input
                current_time = self.sim.data.time
                control_input = control_function(current_time,self.sim)
                
                # Step simulation
                self.sim.step(control_input)

                # write to sim.record_
                self.log_data_step()

                # Render
                self.render()
                # print(f'{current_time}')
                # Log muscle state
                if current_time - last_log_time >= log_interval:
                    self.plot_muscle_activations()
                    self.plot_muscle_activations()
                    last_log_time = current_time
                    
                # Real-time synchronization
                if self.real_time:
                    elapsed = time.time() - start_time
                    sim_time = self.sim.data.time - sim_start_time
                    if sim_time > elapsed:
                        time.sleep(sim_time - elapsed)
            self.transfer_data_to_np()
                    
    def close(self):
        """Close the viewer"""
        if self.viewer:
            self.viewer.close()
            self.is_running = False
import mujoco
import mujoco.viewer
import numpy as np
from typing import Optional, Callable, List
import time
from src.MSK_Model import MusculoskeletalSimulation

class MusculoskeletalVisualizer:
    """Enhanced visualizer for musculoskeletal simulations with muscle activation display"""
    
    def __init__(self, MSK_model, azimuth=0, elevation=0, distance=0,lookat=[0,0,0]):
        """
        Initialize musculoskeletal visualizer
        
        Args:
            MSK_model: MusculoskeletalSimulation instance
        """
        self.sim = MSK_model
        self.viewer = None

        self.real_time = True
        
        # settings
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = lookat

        # Visualization options
        self.show_activation_colors = True
        # self.show_muscle_forces = True
        # self.force_scale = 0.001  # Scale factor for force visualization

        self.scene = mujoco.MjvScene(self.sim.model, maxgeom=1000)
        
    def _viewer_settings(self):
        """
        Setting of model.viewer

        """
        if  self.viewer:

            # frame 
            self.viewer.opt.frame = True
            
            # viewer opt settings
            # self.viewer.opt.geomgroup[1] = False
            self.viewer.opt.sitegroup[2] = True
            # self.viewer.opt.tendongroup[0] = False
                
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0    # Show convex hulls
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1       # Show textures
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1         # Show joints
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1        # Hide cameras
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1      # Hide actuators
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = 1    # Hide activation
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 0         # Hide lights
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1        # Show tendons
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0   # Hide range finders
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 1    # Show constraints
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0      # Hide inertia boxes
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] = 0    # Hide scaled inertia
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0     # Hide perturbation forces
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0       # Hide perturbation object
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # Show contact points
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 1        # Show contact points    
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # Show contact forces
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = 0  # Hide contact splits
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0   # Disable transparency
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = 1   # Auto-connect bodies
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0           # Hide center of mass
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 1        # Show selection
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1        # Show static bodies
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 1          # Show skin

            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = 0      # Hide flex vertices
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = 1      # Show flex edges
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 1      # Show element faces
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 1      # Show flex smooth skin
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = 0       # Hide body bounding volume
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXBVH] = 1       # Show flex bounding volume
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_MESHBVH] = 1       # Show mesh bounding volume 
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SDFITER] = 1       # Show 

            # set focus
            self.viewer.cam.azimuth = self.azimuth      # 
            self.viewer.cam.elevation = self.elevation   # Tilt camera downward
            self.viewer.cam.distance = self.distance    # Zoom out
            self.viewer.cam.lookat = self.lookat   # Center 

    def render(self):
        """Render current frame with muscle activation coloring"""
        
        if self.show_activation_colors:
                self._update_muscle_colors()
        self.viewer.sync()
            
    # @staticmethod
    def draw_site_frame(self, site_names: List[str]=[], AxisLen: float=0.1):
                #    pos:np.ndarray ,xmat:np.ndarray, AxisLen: float=0.1):
        """Draw RGB axis lines at given position and orientation."""
        pass
        colors = [(1, 0, 0, 1),  # X - red
                (0, 1, 0, 1),  # Y - green
                (0, 0, 1, 1)]  # Z - blue
        for name in site_names:
            # check is site existed
            site_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE,name)
            if  site_id == -1:
                raise ValueError(f"Site '{name}' not found.")
                
            pos = self.sim.data.site_xpos[site_id]
            rotm = self.sim.data.site_xmat[site_id].reshape(3, 3)
            for i in range(3):
                # add one user_scn.geom
                self.viewer.user_scn.ngeom += 1
                arrow_to = pos + AxisLen * rotm[:, i]
                mujoco.mjv_initGeom(
                    geom=self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom-1],
                    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=np.array(colors[i])
                )
                mujoco.mjv_connector(
                    geom=self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom-1],
                    type=mujoco.mjtGeom.mjGEOM_ARROW.value,
                    width=0.02*AxisLen,
                    from_=pos,
                    to=arrow_to)



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
        print(f"muscle activation: {activations}")
        print(f"muscle force: {forces}")
        print("---")

    def transfer_data_to_np(self):
        """
        transfer self.sim.record_data to np.ndarray
        """
        for key in self.sim.record_data:
            self.sim.record_data[key] = np.array(self.sim.record_data[key])

    def default_control_function(t: float) -> np.ndarray:
        print('no control function is give')


    def default_log_function(model:MusculoskeletalSimulation):
        """
        log information in terminal 
        """
        pass
        # # template of log_function
        # activations = model.get_muscle_activations()
        # forces = model.get_muscle_forces()
        # site_dis = [np.linalg.norm(model.data.site(i).xpos - model.data.site(i+3).xpos)
        #             for i in range(3)]
        
        # # Print activation summary
        # print(f"Time: {model.data.time:.3f}")
        # print(f"muscle activation: {activations}")
        # print(f"muscle force: {forces}")
        # print(f"site distance: {site_dis}")
        # print("---")

    def default_record_function(model:MusculoskeletalSimulation):
        """
        record simulation data into model.record_data
        """
        pass
        #
        # if len(model.record_data) == 0:
        #     model.record_data = {
        #             "time": [],
        #             "qpos": [],
        #             "qvel": [],
        #             "ctrl": [],
        #             "act" : [],
        #             "mfrc": [],
        #             "site_dis":[]
        #         }

        # site_dis = np.linalg.norm(model.data.site(0).xpos - model.data.site(1).xpos)
        # model.record_data["time"].append(model.data.time)
        # model.record_data["qpos"].append(model.data.qpos.copy())
        # model.record_data["qvel"].append(model.data.qvel.copy())
        # model.record_data["ctrl"].append(model.data.ctrl.copy())
        # model.record_data["act"].append(model.data.act.copy())
        # model.record_data["mfrc"].append(model.data.actuator_force.copy())
        # model.record_data["site_dis"].append(site_dis)

    def run_simulation(self, 
                      control_function: Callable[[float], np.ndarray],
                      log_function: Callable = default_log_function, 
                      record_function: Callable = default_record_function,
                      render_function: Callable = None,
                      duration: float = 10.0,
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
            # set viewer
            self._viewer_settings()
            
            start_time = time.time()
            sim_start_time = self.sim.data.time
            last_log_time = 0
            
            while self.viewer.is_running() and (self.sim.data.time - sim_start_time) < duration:
                # Get control input
                current_time = self.sim.data.time
                control_input = control_function(current_time,self.sim)
                
                # Step simulation
                self.sim.step(control_input)

                # write to sim.record_
                record_function(self.sim)

                # Render
                self.render()
                
                # Log muscle state
                if current_time - last_log_time >= log_interval:
                    log_function(self.sim)
                    last_log_time = current_time
                    
                # Real-time synchronization
                if self.real_time:
                    elapsed = time.time() - start_time
                    sim_time = self.sim.data.time - sim_start_time
                    if sim_time > elapsed:
                        time.sleep(sim_time - elapsed)
            
            # transfer data to numpy array
            self.transfer_data_to_np()
                    
    # def close(self):
    #     """Close the viewer"""
    #     if self.viewer:
    #         self.viewer.close()
    #         self.is_running = False
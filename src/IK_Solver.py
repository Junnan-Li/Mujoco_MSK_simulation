import numpy as np
import mujoco
from enum import Enum
from scipy.spatial.transform import Rotation as R
from typing import Optional
from dataclasses import dataclass, field

from src.MSK_Model import MusculoskeletalSimulation
from src.IKParams import IK_Params, IK_Target, IK_Target_Mode
from src.visualizer import MusculoskeletalVisualizer
import src.utilities as ut 



class IK_Algorithm(Enum):
    Newton_Raphson = 0          
    Gauss_Newton = 1         
    Levenburg_Marquadt = 2        

@dataclass
class IK_results:
    iter: int = 0
    qpos: np.ndarray = field(default_factory=lambda: np.array([])) 
    site_error: np.ndarray = field(default_factory=lambda: np.array([])) 
    status: int = 0

class IK_Solver:
    """ A class for solving inverse kinematic problem of MSK model"""
    def __init__(self, 
                 MSK_model:MusculoskeletalSimulation, 
                 viz: Optional['MusculoskeletalVisualizer'] = None):
        self.MSKmodel = MSK_model

        self.ik_prm = IK_Params()
        self.IK_method = IK_Algorithm.Newton_Raphson

        self.viz = viz

        self.results = IK_results()

    def set_target(self, target):
        self.target = target # target only has site_targets entity
        self.site_ids_targets_list = [
            (mujoco.mj_name2id(self.MSKmodel.model, mujoco.mjtObj.mjOBJ_SITE, name), target)
            for name, target in self.target.site_targets.items()]
        self.nsite = len(self.site_ids_targets_list)


    def get_site_Jac(self) -> np.ndarray:
        """calcualte and stack the Jacobian of target sites in terms of all joints.
            
            Returns:
            jacobian: np.ndarray
                If mode = trans_only, rot_only: shape ( 3*nsite, nv)
                Else: shape (6*nsite,  nv), stacked position + rotation Jacobian
        """

        if self.ik_prm.target_type == IK_Target_Mode.trans_only.value or self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
            jacobian = np.zeros((3*self.nsite, self.MSKmodel.model.nv))
        else:
            jacobian = np.zeros((6*self.nsite, self.MSKmodel.model.nv))
    
        for i,(site_id, _) in enumerate(self.site_ids_targets_list):
            jacp = np.zeros((3, self.MSKmodel.model.nv))  
            jacr = np.zeros((3, self.MSKmodel.model.nv))  
            mujoco.mj_jacSite(self.MSKmodel.model, self.MSKmodel.data, jacp, jacr, site_id) 
            if self.ik_prm.target_type == IK_Target_Mode.trans_only.value:
                jacobian[3*i:3*i+3,:] = jacp
            elif self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
                jacobian[3*i:3*i+3,:] = jacr
            else:
                jacobian[6*i:6*i+6,:] = np.vstack([jacp, jacr])
        return jacobian
    
    def get_current_coord(self) -> np.ndarray:
        """read the position/orientation of target sites
         pos: (N,3) if translation/rotation-only, or (N,6) if including rotation error 
        """
        site_id = [x[0] for x in self.site_ids_targets_list]
        
        if self.ik_prm.target_type == IK_Target_Mode.trans_only.value:
            current_pos = self.MSKmodel.data.site_xpos[site_id] # (nsite,3)
            return current_pos
        elif self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
            site_rotm = self.MSKmodel.data.site_xmat[site_id].reshape(self.nsite,3, 3) # 
            current_rotv = R.from_matrix(site_rotm).as_rotvec() # (nsite,3)
            return current_rotv
        else:
            current_pos = self.MSKmodel.data.site_xpos[site_id] # (nsite,3)
            site_rotm = self.MSKmodel.data.site_xmat[site_id].reshape(self.nsite,3, 3) # 
            current_rotv = R.from_matrix(site_rotm).as_rotvec() # (nsite,3)
            return np.hstack([current_pos,current_rotv])
        
    def Rotmat_log(self, mat):
        #print('rotmatlog ', mat)
        if np.allclose(mat, np.eye(mat.shape[0])): # check if its an identity
            return np.array([0, 0, 0])
        theta = np.arccos( (np.trace(mat)-1) / 2 )
        n = 1 / (2*np.sin(theta)) * np.array([[mat[2, 1] - mat[1, 2]], [mat[0, 2] - mat[2, 0]], [mat[1, 0] - mat[0, 1]]])
        return (theta * n).reshape(1, 3) 
    
    def calc_rot_err(self, target_coord, current_coord):
        num_targets = len(self.target.site_targets)
        err = np.zeros([num_targets, 3])
        for t in range(num_targets):
            current_rotm = R.from_rotvec(current_coord[t, :]).as_matrix().squeeze()
            target_rotm = R.from_euler('xyz', target_coord[t, :]).as_matrix().squeeze()
            err[t, :] = self.Rotmat_log(target_rotm @ current_rotm.T)
        return err


    def cal_error(self) -> np.ndarray:
        """calculate the position/orientation error of sites
         error: (N,3) if translation/rotation-only, or (N,6) if including rotation error
        """
        # site_id = [x[0] for x in self.site_ids_targets_list]
        current_coord = self.get_current_coord() # (nsite,3)
        target_coord = np.stack([x[1] for x in self.site_ids_targets_list])
        if self.ik_prm.target_type == IK_Target_Mode.trans_only.value:
            return target_coord - current_coord
        elif self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
            return self.calc_rot_err(target_coord, current_coord)
        else:
            error_rotv = self.calc_rot_err(target_coord[:,3:], current_coord[:,3:])
            # error_rotv = R.from_matrix(error_rotm.as_matrix()).as_rotvec()

            error_6d = np.hstack([target_coord[:,:3]-current_coord[:,:3], error_rotv])
            return error_6d

    # def cal_error(self) -> np.ndarray:
    #     """calculate the position/orientation error of sites
    #      error: (N,3) if translation-only, or (N,6) if including rotation error
    #     """
    #     # site_id = [x[0] for x in self.site_ids_targets_list]
    #     current_pos = self.get_current_coord() # (nsite,3)
    #     target_pos = np.stack([x[1] for x in self.site_ids_targets_list])
    #     if self.ik_prm.target_type == IK_Target_Mode.trans_only.value:
    #         return target_pos - current_pos
    #     else:
    #         current_rotv = current_pos[:,3:]
    #         target_rotv = target_pos[:,3:]
    #         current_rotm = R.from_rotvec(current_rotv)
    #         target_rotm = R.from_euler('xyz', target_rotv)

    #         error_rotv = (target_rotm * current_rotm.inv()).as_rotvec()
    #         # error_rotv = R.from_matrix(error_rotm.as_matrix()).as_rotvec()

    #         error_6d = np.hstack([target_pos[:,:3]-current_pos[:,:3], error_rotv])
    #         return error_6d

    def solve(self):

        qpos_original = self.MSKmodel.data.qpos.copy()

        for iter in range(self.ik_prm.max_iter):
            
            # mujoco.mj_fwdPosition(self.model, self.data)
            # mujoco.mj_forward(self.MSKmodel.model, self.MSKmodel.data)
            mujoco.mj_step1(self.MSKmodel.model, self.MSKmodel.data)

            error = self.cal_error() # 
            #print(f"iter: {iter}  error: {error}")
            
            end_flag = False
            if self.ik_prm.target_type == IK_Target_Mode.trans_only.value:
                if np.linalg.norm(error) < self.ik_prm.tol_pos:
                    end_flag = True
            elif self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
                if np.linalg.norm(error) < self.ik_prm.tol_rot:
                    end_flag = True
            elif self.ik_prm.target_type == IK_Target_Mode.trans_rot.value:
                if np.linalg.norm(error[:,:3]) < self.ik_prm.tol_pos and np.linalg.norm(error[:,3:]) < self.ik_prm.tol_rot:
                    end_flag = True

            if end_flag == True:
                self.results.iter = iter
                self.results.qpos = self.MSKmodel.data.qpos.copy()
                self.results.site_error = error
                self.results.status = 1
                return 
            
            error_cat = np.concatenate(error)
            jac = self.get_site_Jac()
            
            match self.IK_method.value:
                case 0: # Newton-Raphson
                    dq = np.linalg.pinv(jac) @ error_cat
                    qpos_new = self.MSKmodel.data.qpos[:] + dq 
                    self.MSKmodel.data.qpos[:] = np.clip(qpos_new,self.MSKmodel.model.jnt_range[:,0],self.MSKmodel.model.jnt_range[:,1])
                    self.MSKmodel.data.qvel[:] = 0 
                case 1: # Gauss-Newton
                    pass
                case 2: # Levenburg-Marquadt
                    if self.ik_prm.target_type == IK_Target_Mode.trans_only.value or self.ik_prm.target_type == IK_Target_Mode.rot_only.value:
                        LM_w_e_all = np.eye(3*self.nsite) * 1e-1 # 6nx6n
                    else:
                        LM_w_e_all = np.eye(6*self.nsite) * 1e-1 # 6nx6n
                    g_i = jac.transpose() @ LM_w_e_all @ error_cat
                    dq = np.linalg.pinv(jac.transpose() @ LM_w_e_all @ jac + self.ik_prm.LM_w_d*np.eye(self.MSKmodel.model.nv)) @ g_i
                    qpos_new = self.MSKmodel.data.qpos[:] + dq
                    self.MSKmodel.data.qpos[:] = np.clip(qpos_new,self.MSKmodel.model.jnt_range[:,0],self.MSKmodel.model.jnt_range[:,1])
                    self.MSKmodel.data.qvel[:] = 0 

            #self.viz.draw_site_frame([name for name in self.target.site_targets.keys()])            
            #self.viz.render()
        #print(error_cat)
        self.results.iter = iter
        self.results.qpos = self.MSKmodel.data.qpos.copy()
        self.results.site_error = error
        self.results.status = 10 

        

            



            





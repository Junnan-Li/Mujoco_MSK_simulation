import numpy as np
import mujoco
from enum import Enum
from scipy.spatial.transform import Rotation as R

from src.MSK_Model import MusculoskeletalSimulation
from src.IKParams import IK_Params, IK_Target


class IK_Algorithm(Enum):
    Newton_Raphson = 0          
    Gauss_Newton = 1         
    Levenburg_Marquadt = 2        

class IK_Solver:
    """ A class for solving inverse kinematic problem of MSK model"""
    def __init__(self, MSK_model:MusculoskeletalSimulation, target: IK_Target):
        self.MSKmodel = MSK_model

        self.ik_prm = IK_Params()
        self.IK_method = IK_Algorithm.Levenburg_Marquadt

        self.target = target # target only has site_targets entity
        self.site_ids_targets_list = [
            (mujoco.mj_name2id(self.MSKmodel.model, mujoco.mjtObj.mjOBJ_SITE, name), target)
            for name, target in self.target.site_targets.items()]
        self.nsite = len(self.site_ids_targets_list)

    def get_site_Jac(self) -> np.ndarray:
        """calcualte and stack the Jacobian of target sites in terms of all joints.
            
            Returns:
            jacobian: np.ndarray
                If trans_only is True: shape ( 3*nsite, nv)
                Else: shape (6*nsite,  nv), stacked position + rotation Jacobian
        """

        if self.ik_prm.trans_only:
            jacobian = np.zeros((3*self.nsite, self.MSKmodel.model.nv))
        else:
            jacobian = np.zeros((6*self.nsite, self.MSKmodel.model.nv))
    
        for i,(site_id, _) in enumerate(self.site_ids_targets_list):
            jacp = np.zeros((3, self.MSKmodel.model.nv))  
            jacr = np.zeros((3, self.MSKmodel.model.nv))  
            mujoco.mj_jacSite(self.MSKmodel.model, self.MSKmodel.data, jacp, jacr, site_id) 
            if self.ik_prm.trans_only:
                jacobian[3*i:3*i+3,:] = jacp
            else:
                jacobian[6*i:6*i+6,:] = np.vstack([jacp, jacr])
        return jacobian
    
    def get_current_target_pos(self) -> np.ndarray:
        """read the position/orientation of target sites
         pos: (N,3) if translation-only, or (N,6) if including rotation error 
        """
        site_id = [x[0] for x in self.site_ids_targets_list]
        current_pos = self.MSKmodel.data.site_xpos[site_id] # (nsite,3)
        if self.ik_prm.trans_only:
            return current_pos
        else:
            site_rotm = self.MSKmodel.data.site_xmat[site_id].reshape(self.nsite,3, 3) # 
            current_rotv = R.from_matrix(site_rotm).as_rotvec() # (nsite,3)

            return np.hstack([current_pos,current_rotv])


    def cal_error(self) -> np.ndarray:
        """calculate the position/orientation error of sites
         error: (N,3) if translation-only, or (N,6) if including rotation error
        """
        # site_id = [x[0] for x in self.site_ids_targets_list]
        current_pos = self.get_current_target_pos() # (nsite,3)
        target_pos = np.stack([x[1] for x in self.site_ids_targets_list])
        if self.ik_prm.trans_only:
            return target_pos - current_pos
        else:
            current_rotv = current_pos[:,3:]
            target_rotv = target_pos[:,3:]
            current_rotm = R.from_rotvec(current_rotv)
            target_rotm = R.from_euler('xyz', target_rotv)

            error_rotv = (target_rotm * current_rotm.inv()).as_rotvec()
            # error_rotv = R.from_matrix(error_rotm.as_matrix()).as_rotvec()

            error_6d = np.hstack([target_pos[:,:3]-current_pos[:,:3], error_rotv])
            return error_6d



    def solve(self):

        qpos_original = self.MSKmodel.data.qpos.copy()

        for iter in range(self.ik_prm.max_iter):
            
            # mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_forward(self.MSKmodel.model, self.MSKmodel.data)

            error = self.cal_error() # 

            if np.linalg.norm(error[:,:3]) < self.ik_prm.tol_pos and np.linalg.norm(error[:,3:]) < self.ik_prm.tol_rot: # TODO
                return True
            error_cat = np.concatenate(error)
            jac = self.get_site_Jac
            
            match self.IK_method:
                case 0: # Newton-Raphson
                    pass
                case 1: # Gauss-Newton
                    pass
                case 2: # Levenburg-Marquadt
                    pass

            



            





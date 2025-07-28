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


    def get_site_Jac(self) -> np.ndarray:
        """calcualte and stack the Jacobian of target sites in terms of all joints.
            
            Returns:
            jacobian: np.ndarray
                If trans_only is True: shape (nsite, 3, nv)
                Else: shape (nsite, 6, nv), stacked position + rotation Jacobian
        """

        nsite = len(self.site_ids_targets_list)
        if self.ik_prm.trans_only:
            jacobian = np.zeros((nsite, 3, self.MSKmodel.model.nv))
        else:
            jacobian = np.zeros((nsite, 6, self.MSKmodel.model.nv))
    
        for i,(site_id, _) in self.site_ids_targets_list:
            jacp = np.zeros((3, self.MSKmodel.model.nv))  
            jacr = np.zeros((3, self.MSKmodel.model.nv))  
            mujoco.mj_jacSite(self.MSKmodel.model, self.MSKmodel.data, jacp, jacr, site_id) 
            if self.ik_prm.trans_only:
                jacobian[i] = jacp
            else:
                jacobian[i] = np.vstack([jacp, jacr])
        return jacobian
    

    def cal_error(self) -> np.ndarray:
        """calculate the position/orientation error of sites
         error: (N, 3) if translation-only, or (N, 6) if including rotation error
        """
        site_id = [x[0] for x in self.site_ids_targets_list]
        current_pos = self.MSKmodel.data.site_xpos[site_id]
        target_pos = np.stack([x[1] for x in self.site_ids_targets_list])
        if self.ik_prm.tran_only:
            return target_pos - current_pos
        else:
            error_list = []
            for i, id in enumerate(site_id):
                error_pos = target_pos[i,:3] - current_pos[i]
                site_rotm = self.data.site_xmat[id].reshape(3, 3)
                target_rotm = R.from_euler('xyz', target_pos[i,3:6]).as_matrix()
                error_rotm = target_rotm @ site_rotm.T
                error_rotv = R.from_matrix(error_rotm).as_rotvec()
                error_i = np.concatenate([error_pos, error_rotv])
                error_list.append(error_i)
            return np.vstack(error_list)



    def solve(self):

        qpos_original = self.MSKmodel.data.qpos.copy()

        for iter in range(self.ik_prm.max_iter):
            
            # mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_forward(self.MSKmodel.model, self.MSKmodel.data)

            error = self.cal_error

            if np.linalg.norm(error) < self.ik_prm.tol_pos: # TODO
                pass
                return True
            
            jac = self.get_site_Jac
            
            match self.IK_method:
                case 0: # Newton-Raphson
                    pass
                case 1: # Gauss-Newton
                    pass
                case 2: # Levenburg-Marquadt
                    pass

            



            





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
            (mujoco.mj_name2id(self.model.model, mujoco.mjtObj.mjOBJ_SITE, name), target)
            for name, target in self.target.site_targets]
        


    def get_site_Jac(self) -> np.ndarray:
        """calcualte and stack the Jacobian of target sites in terms of all joints.
        
            IK_Solver.ik_prm.trans_only: [3 x nsite, nv]
        """
        jac = np.empty((6,self.MSKmodel.model.nv))
        jac = []
        for site_id, _ in self.site_ids_targets_list:
            jacp = np.zeros((3, self.MSKmodel.model.nv))  
            jacr = np.zeros((3, self.MSKmodel.model.nv))  
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id) 
            if self.ik_prm.trans_only:
                jac.append(np.vstack(jacp))
            else:
                jac.append(np.vstack([jacp,jacr]))
        return jac
    
    def cal_error(self) -> np.ndarray:
        """calculate the position/orientation error of sites"""
        pass



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

            



            





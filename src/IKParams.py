from dataclasses import dataclass
import numpy as np


@dataclass
class IK_Params:
    
    tran_only: bool = False
    max_iter: int = 1000
    tol_pos: float = 1e-4
    tol_rot: float = 1e-3




class IK_Target:
    target_pos: np.ndarray
    target_quat: np.ndarray

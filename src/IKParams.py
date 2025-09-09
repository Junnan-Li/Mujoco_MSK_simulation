from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass
class IK_Params:
    
    trans_only: bool = False
    max_iter: int = 1000
    tol_pos: float = 1e-4
    tol_rot: float = 1e-3
    LM_w_d: float = 1e-6



class IK_Target:
    # 6 dimensional vector [pos ori]
    # translational position
    # euler angle xyz as rotation 
    site_targets: Dict[str,np.ndarray] = field(default_factory=dict)


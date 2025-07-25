from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass
class IK_Params:
    
    tran_only: bool = False
    max_iter: int = 1000
    tol_pos: float = 1e-4
    tol_rot: float = 1e-3




class IK_Target:
    site_targets: Dict[str,np.ndarray] = field(default_factory=dict)


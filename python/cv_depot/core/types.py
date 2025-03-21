from typing import Optional, Union
from pathlib import Path

import numpy as np

from cv_depot.core.enum import Anchor
from cv_depot.core.color import BasicColor, Color
# ------------------------------------------------------------------------------

AnyAnchor = Union[Anchor, str]
AnyColor = Union[Color, BasicColor, str]
Filepath = Union[str, Path]
OptArray = Optional[np.ndarray]

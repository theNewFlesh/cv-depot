from typing import Union
from pathlib import Path

from cv_depot.core.enum import Anchor
from cv_depot.core.color import BasicColor, Color
# ------------------------------------------------------------------------------

AnyAnchor = Union[Anchor, str]
AnyColor = Union[Color, BasicColor, str]
Filepath = Union[str, Path]

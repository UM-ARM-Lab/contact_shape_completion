from dataclasses import dataclass
from typing import Type

from contact_shape_completion import scenes


@dataclass()
class EvaluationDetails:
    scene_type: Type[scenes.Scene]
    network: str
    method: str

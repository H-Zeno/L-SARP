from enum import Enum
from typing import List, Optional
from dataclasses import dataclass


DEFAULT_RETRIEVAL_PLUGINS = ["nav", "text", "sql", "image"]


@dataclass
class SceneConfig:
    name: str
    retrieval_plugins: Optional[List[str]] = None

    def __post_init__(self):
        if self.retrieval_plugins is None:
            self.retrieval_plugins = DEFAULT_RETRIEVAL_PLUGINS


class Scene(Enum):
    TEST_VIRTUAL_SCENE = SceneConfig("test_virtual_scene", ["nav", "text"])
    SEMANTIC_CORNER = SceneConfig("semantic_corner")
    SEMANTIC_CORNER_WITH_BED = SceneConfig("semantic_corner_with_bed")
    APARTMENT_0 = SceneConfig("apartment_0", ["text"])
    APARTMENT_1 = SceneConfig("apartment_1")
    APARTMENT_2 = SceneConfig("apartment_2")
    ROOM_0 = SceneConfig("room_0")
    ROOM_1 = SceneConfig("room_1")
    ROOM_2 = SceneConfig("room_2")
    OFFICE_0 = SceneConfig("office_0")
    OFFICE_2 = SceneConfig("office_2")
    OFFICE_3 = SceneConfig("office_3")
    OFFICE_4 = SceneConfig("office_4")
    HOTEL_0 = SceneConfig("hotel_0")
    FRL_APARTMENT_0 = SceneConfig("frl_apartment_0")
    FRL_APARTMENT_1 = SceneConfig("frl_a`partment_1")

    @property
    def value(self) -> str:
        return self._value_.name

    @property 
    def retrieval_plugins(self) -> List[str]:
        return self._value_.retrieval_plugins


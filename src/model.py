from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import TypedDict

from numpy import float64 as np_float64
from numpy.typing import NDArray

type FloatVector = NDArray[np_float64]
type MemoryRow = tuple[FloatVector, int, float, FloatVector]

@dataclass(frozen=True)
class Building:
    "Represents a building that generates atoms"
    name: str
    level: int
    current_rate_per_sec: float
    upgrade_price: int
    is_available: bool

class UpgradeType(IntEnum):
    "Represents the target of an upgrade"
    BUILDING_MOLECULE = 0
    BUILDING_CRYSTAL = 1
    BUILDING_NANOSTRUCTURE = 2
    BUILDING_MICROORGANISM = 3
    BUILDIN_ROCK = 4
    BUILDING_PLANET = 5
    BUILDING_STAR = 6
    BUIDLING_NEUTRONSTAR = 7
    BUILDING_BLACKHOLE = 8
    CLICK_POWER_MUL = 9
    CLICK_POWER_VAL = 10
    CLICK_POWER_APS = 11
    GLOBAL_BOOST = 12
    GLOBAL_ACHIEVEMENTS_MUL = 13
    POWERUP_INTERVAL = 14
    LEVEL_BOOST = 15
    UNLOCK_LEVELS = 16

@dataclass(frozen=True)
class Upgrade:
    "Represents an upgrade"
    name: str
    target: UpgradeType
    level: int
    price: int
    is_available: bool

@dataclass(frozen=True)
class GameState:
    "Represents a snapshot of the game state"
    atoms_count: int
    rate_per_sec: float
    available_powerup: bool
    buildings: list[Building]
    upgrades: list[Upgrade]

    @property
    def available_actions(self) -> list[str]:
        "Return a list of available actions"
        available_actions = [DefaultActions.WAIT.value, DefaultActions.CLICK_CENTER.value]
        if self.available_powerup:
            available_actions.append(DefaultActions.CLICK_POWERUP.value)
        for i, building in enumerate(self.buildings):
            if building.is_available:
                available_actions.append(f"build_{i}")
        for i, upgrade in enumerate(self.upgrades):
            if upgrade.is_available:
                available_actions.append(f"upgrade_{i}")
        if any(action not in ACTION_SPACE for action in available_actions):
            raise ValueError(
                "Some available actions are not in the action space",
                available_actions
            )
        return available_actions

@dataclass(frozen=True)
class TrainingRecord:
    "Contains stats about a training session"
    duration: float
    samples: int
    loss: float

    def to_dict(self):
        "Returns a dictionary representation of the record"
        return asdict(self)

class LocalStorage(TypedDict):
    "Represents the local storage of the game"
    achievements: list[str]
    atoms: int
    protons: int
    skillUpgrades: list[str]
    totalClicks: int
    totalXP: int
    upgrades: list[str]
    totalProtonises: int
    version: int


MAX_UPGRADES_COUNT = 10
MAX_BUILDINGS_COUNT = 10

class DefaultActions(StrEnum):
    "Default actions"
    WAIT = "wait"
    CLICK_CENTER = "click_center"
    CLICK_POWERUP = "click_powerup"

ACTION_SPACE: dict[str, int] = {
    DefaultActions.WAIT: 0,
    DefaultActions.CLICK_CENTER: 1,
    DefaultActions.CLICK_POWERUP: 2,
}
for upgrade_id in range(MAX_UPGRADES_COUNT):
    ACTION_SPACE[f"upgrade_{upgrade_id}"] = max(ACTION_SPACE.values()) + 1
for building_id in range(MAX_BUILDINGS_COUNT):
    ACTION_SPACE[f"build_{building_id}"] = max(ACTION_SPACE.values()) + 1

def get_action_name_from_id(action_id: int) -> str:
    "Return the action name from its id"
    for action_name, id_ in ACTION_SPACE.items():
        if id_ == action_id:
            return action_name
    raise ValueError("Unknown action id", action_id)

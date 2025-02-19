import csv
import json
import time
from dataclasses import asdict, dataclass
from uuid import uuid4

import numpy as np

from .model import DefaultActions, GameState, LocalStorage, MemoryRow, TrainingRecord

_ACTION_LOGS_FILE = "actions.csv"
_TRAINING_LOGS_FILE = "training.csv"
_MEMORY_LOGS_FILE = "memory.csv"
_RESULTS_LOGS_FILE = "finals.csv"

_TRAINING_FOLDER = "training"
_PROD_FOLDER = "prod"

class DataRecorder:
    "Record system performances and history data"

    @dataclass(frozen=True)
    class GameSnapshot:
        "Represents a simplified snapshot of the game state"
        atoms_count: int
        rate_per_sec: float
        available_actions: list[str]
        chosen_action: str
        reward: float | None

        def to_dict(self):
            "Returns a dictionary representation of the snapshot"
            return asdict(self)

    @dataclass(frozen=True)
    class FinalResult:
        "Contains stats about a model result when time is up"
        time_spent: float
        steps_count: int
        memory_size: int
        atoms_count: int
        rate_per_sec: float
        total_clicks: int | None
        upgrades_count: int | None
        unique_buildings_count: int
        buildings_sum: int
        max_building: str | None

        def to_dict(self):
            "Returns a dictionary representation of the record"
            return asdict(self)

    def __init__(self, *, prod: bool = False):
        self._folder = _PROD_FOLDER if prod else _TRAINING_FOLDER
        self.action_log_file = f"{self._folder}/{_ACTION_LOGS_FILE}"
        self.training_log_file = f"{self._folder}/{_TRAINING_LOGS_FILE}"
        self.memory_log_file = f"{self._folder}/{_MEMORY_LOGS_FILE}"
        self.results_log_file = f"{self._folder}/{_RESULTS_LOGS_FILE}"

        self.time_per_action: list[float] = []
        self.last_action_start: float | None = None
        self.snapshots: dict[float, DataRecorder.GameSnapshot] = {}
        self.training_records: dict[float, TrainingRecord] = {}
        self.uuid = uuid4().hex

    @property
    def actions_count(self) -> int:
        "Return the number of actions"
        return len(self.time_per_action)

    @property
    def start_time(self) -> float:
        "Return the start time of the experiment"
        try:
            return list(self.snapshots.keys())[0]
        except IndexError:
            return 0

    def on_action_start(self):
        "Start a new action chrono"
        self.last_action_start = time.time()

    def on_action_stop(self, game_state: GameState, action_id: str, reward: float | None):
        "Stop the current action chrono"
        now = time.time()
        if self.last_action_start is None:
            print("WARNING: No action started.")
        else:
            self.time_per_action.append(now - self.last_action_start)
            self.last_action_start = None
        snapshot = DataRecorder.GameSnapshot(
            game_state.atoms_count,
            game_state.rate_per_sec,
            game_state.available_actions,
            self._get_action_name(game_state, action_id),
            reward,
        )
        self.snapshots[now] = snapshot
        self._append_action_log(now, snapshot)

    def record_training(self, record: TrainingRecord):
        "Record a training session"
        now = time.time()
        self.training_records[now] = record
        self._append_training_log(now, record)

    def record_memory(self, memory_row: MemoryRow):
        "Record a memory row"
        now = time.time()
        self._append_memory_log(now, memory_row)

    def record_final_result(self, game_state: GameState, memory_size: int,
                            local_storage: LocalStorage | None):
        "Record the final result of an agent"
        now = time.time()
        upgraded_buildings = [b for b in game_state.buildings if b.level > 0]
        result = DataRecorder.FinalResult(
            time_spent=round(now - self.start_time, 3),
            steps_count=len(self.snapshots),
            memory_size=memory_size,
            atoms_count=game_state.atoms_count,
            rate_per_sec=game_state.rate_per_sec,
            total_clicks=local_storage["totalClicks"] if local_storage else None,
            upgrades_count=len(local_storage["upgrades"]) if local_storage else None,
            unique_buildings_count=len([b for b in game_state.buildings if b.level > 0]),
            buildings_sum=sum(b.level for b in game_state.buildings),
            max_building=max(
                upgraded_buildings,
                key=lambda b: b.level
            ).name if upgraded_buildings else None,
        )
        self._append_final_result(now, result)

    def get_average_time_per_action(self) -> float:
        "Return the average time per action"
        if len(self.time_per_action) == 0:
            return 0.0
        return sum(self.time_per_action) / len(self.time_per_action)

    def get_time_since_last_active_action(self) -> float:
        "Return the time since the last non-wait action"
        reverse_snapshot_times = list(self.snapshots.keys())[::-1]
        for snapshot_time in reverse_snapshot_times:
            if self.snapshots[snapshot_time].chosen_action != DefaultActions.WAIT:
                return time.time() - snapshot_time
        return 0.0

    def get_memory_from_file(self) -> list[MemoryRow]:
        "Return the memory rows from the memory file"
        try:
            with open(self.memory_log_file, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                memory = [
                    (
                        np.array(json.loads(row["state"])),
                        int(row["action"]),
                        float(row["cumulative_reward"]),
                        np.array(json.loads(row["state"])),
                    )
                    for row in reader
                ]
            print(f"Loaded {len(memory)} memories from {self._folder} file")
            return memory
        except FileNotFoundError:
            return []

    def _get_action_name(self, game_state: GameState, action_id: str) -> str:
        if action_id.startswith("upgrade_"):
            index = int(action_id.split("_")[1])
            if index >= len(game_state.upgrades):
                return action_id
            return "Upgrade: " + " ".join(game_state.upgrades[index].name.split()[:-1])
        if action_id.startswith("build_"):
            index = int(action_id.split("_")[1])
            if index >= len(game_state.buildings):
                return action_id
            return "Building: " + game_state.buildings[index].name
        return action_id

    def _append_action_log(self, timestamp: float, snapshot: "GameSnapshot"):
        dict_snapshot = snapshot.to_dict()
        fieldnames = ["uuid", "relative_timestamp"] + list(dict_snapshot.keys())
        with open(self.action_log_file, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "uuid": self.uuid,
                "relative_timestamp": timestamp - self.start_time,
                **dict_snapshot
            })

    def _append_training_log(self, timestamp: float, record: TrainingRecord):
        dict_record = record.to_dict()
        fieldnames = ["uuid", "relative_timestamp"] + list(dict_record.keys())
        with open(self.training_log_file, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "uuid": self.uuid,
                "relative_timestamp": timestamp - self.start_time,
                **dict_record
            })

    def _append_memory_log(self, timestamp: float, memory_row: MemoryRow):
        dict_row = {
            "uuid": self.uuid,
            "relative_timestamp": timestamp - self.start_time,
            "state": memory_row[0].tolist(),
            "action": memory_row[1],
            "cumulative_reward": memory_row[2],
            "next_state": memory_row[3].tolist(),
        }
        fieldnames = list(dict_row.keys())
        with open(self.memory_log_file, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(dict_row)

    def _append_final_result(self, timestamp: float, result: FinalResult):
        dict_result = result.to_dict()
        fieldnames = ["uuid", "timestamp"] + list(dict_result.keys())
        with open(self.results_log_file, "a+", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({"uuid": self.uuid, "timestamp": timestamp, **dict_result})


# training/actions.csv
#  uuid | relative_timestamp | atoms_count | rate_per_sec | available_actions | chosen_action | reward
#
# training/memory.csv
#  uuid | relative_timestamp | initial_state | action | cumulative_reward | next_state
#
# training/training.csv
#  uuid | relative_timestamp | duration | samples | loss
#
# training/finals.csv
#  uuid | timestamp | time_spent | steps_count | memory_size | atoms_count | rate_per_sec

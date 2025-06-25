import random
import time
from collections import deque

import numpy as np
from keras import Sequential, layers, optimizers
from termcolor import cprint

from .console import ProgressMonitor
from .model import (ACTION_SPACE, MAX_BUILDINGS_COUNT, MAX_UPGRADES_COUNT,
                    Building, DefaultActions, FloatVector, GameState,
                    MemoryRow, TrainingRecord, Upgrade,
                    get_action_name_from_id)


class DQNAgent:
    """Deep Q-Network that should hopefully learn something"""

    def __init__(self, *, prod: bool):
        self.prod = prod
        # 3 for last action, atoms count and rate per sec
        self.state_size = len(ACTION_SPACE) + 3 + MAX_BUILDINGS_COUNT * 4 + MAX_UPGRADES_COUNT * 4
        self.action_size = len(ACTION_SPACE)
        self.memory: deque[MemoryRow] = deque(maxlen=200_000 if prod else 80_000)
        self.max_batch_size = 200_000 if prod else 60_000
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.05 if self.prod else 0.15
        self.epsilon_decay = 0.97 if prod else 0.98
        self.learning_rate = 0.001
        self.n_step = 15 # Number of steps to consider for the training
        self.recent_experiences: deque[MemoryRow] = deque(maxlen=self.n_step)
        self.model = self._build_model()
        self.last_action: int = -len(ACTION_SPACE)
        self.last_state_vector: FloatVector | None = None
        self.has_stopped = False
        self.progress_monitor = ProgressMonitor(list(ACTION_SPACE.keys()))

    def _build_model(self):
        "Build the neural network model"
        cprint("Building the DQN model...", "grey")
        model = Sequential()
        # model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Input(shape=(self.state_size,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        opt = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=opt) # type: ignore
        return model

    def act(self, game_state: GameState) -> str:
        "Normalize the game state and return the best action"
        state_vector, available_actions_vector = self._get_normalized_state_vector(game_state)
        chosen_action_index = self._act(state_vector, available_actions_vector)
        self.last_action = chosen_action_index
        action_name = get_action_name_from_id(chosen_action_index)
        if (
            action_name in game_state.available_actions
            and action_name not in {DefaultActions.WAIT, DefaultActions.CLICK_CENTER}
        ):
            self.progress_monitor.last_action = action_name
        return action_name

    def remember(self, new_game_state: GameState) -> float | None:
        "Remember how the previous action transitionned the game state and how it was rewarded"
        new_state_vector, _ = self._get_normalized_state_vector(new_game_state)
        if self.last_state_vector is None:
            self.last_state_vector = new_state_vector
            return None
        reward = self._calculate_reward(new_game_state)
        self._remember(self.last_state_vector, self.last_action, reward, new_state_vector)
        self.last_state_vector = new_state_vector
        return reward

    def train(self) -> TrainingRecord | None:
        "Train the model on the remembered transitions"
        memory_size = len(self.memory)
        if memory_size < 30:
            cprint("Not enough samples to train the model", "light_green")
            return None
        if memory_size < 100:
            batch_size = memory_size
        else:
            batch_size = min(
                random.randint(max(100, int(memory_size * 2/3)), memory_size),
                self.max_batch_size
            )
        start = time.time()
        loss = self._replay(batch_size)
        training_time = time.time() - start
        self.progress_monitor.last_training_text = (
            f"Training in {training_time:.3f}s using {batch_size} samples. "\
            f"Final loss: {loss:.2e} - epsilon: {self.epsilon:.4f}"
        )
        return TrainingRecord(training_time, batch_size, loss)

    def stop(self):
        "Stop the agent"
        self.has_stopped = True

    def _get_normalized_state_vector(
            self, game_state: GameState) -> tuple[FloatVector, FloatVector]:
        "Compute the normalized state vector and the available actions vector"
        last_action_normalized = self.last_action / len(ACTION_SPACE)
        available_actions_vector = self._get_available_actions_vector(game_state)
        atoms_count_normalized = self._normalize_atoms_count(game_state.atoms_count)
        rate_per_sec_normalized = self._normalize_atoms_count(game_state.rate_per_sec)
        building_vector = self._get_building_vector(game_state.buildings)
        upgrade_vector = self._get_upgrade_vector(game_state.upgrades)
        normalized_state = np.concatenate([
            [last_action_normalized, atoms_count_normalized, rate_per_sec_normalized],
            building_vector,
            upgrade_vector,
            available_actions_vector,
        ])
        # Reshape the input to match the model's input
        return np.reshape(normalized_state, [1, self.state_size]), available_actions_vector

    def _get_available_actions_vector(self, game_state: GameState) -> FloatVector:
        "Return a vector of available actions"
        available_actions = np.zeros(self.action_size)
        for action_name, action_id in ACTION_SPACE.items():
            is_wait_in_first_action = action_name == DefaultActions.WAIT and self.last_action < 0
            if action_name in game_state.available_actions and not is_wait_in_first_action:
                available_actions[action_id] = 1
        return available_actions

    def _normalize_atoms_count(self, atoms_count: float) -> float:
        value = normalize_large_number(atoms_count)
        if value < 0 or value > 1:
            raise ValueError("Atoms count exceeds the expected range", atoms_count)
        return value

    def _get_building_vector(self, buildings: list[Building]) -> FloatVector:
        "Return a vector of building data"
        building_vector = np.zeros((MAX_BUILDINGS_COUNT, 4))
        for i, building in enumerate(buildings):
            building_vector[i, 0] = 1 if building.is_available else 0
            building_vector[i, 1] = normalize_large_number(building.upgrade_price)
            building_vector[i, 2] = normalize_large_number(building.current_rate_per_sec)
            building_vector[i, 3] = normalize_large_number(building.level)
        return building_vector.flatten()

    def _get_upgrade_vector(self, upgrades: list[Upgrade]) -> FloatVector:
        "Return a vector of upgrade data"
        upgrade_vector = np.zeros((MAX_UPGRADES_COUNT, 4))
        for i, upgrade in enumerate(upgrades):
            upgrade_vector[i, 0] = 1 if upgrade.is_available else 0
            upgrade_vector[i, 1] = normalize_large_number(upgrade.price)
            upgrade_vector[i, 2] = upgrade.level / 40
            upgrade_vector[i, 3] = upgrade.target / 30
        return upgrade_vector.flatten()

    def _calculate_reward(self, new_game_state: GameState) -> float:
        "Calculate the immediate reward for the last transition"
        if self.last_state_vector is None:
            return 0
        is_waiting = self.last_action == ACTION_SPACE[DefaultActions.WAIT]
        prev_atoms_count = unnormalize_large_number(self.last_state_vector[0, 1])
        prev_rate_per_sec = unnormalize_large_number(self.last_state_vector[0, 2])
        atoms_diff = new_game_state.atoms_count - prev_atoms_count
        rate_diff = new_game_state.rate_per_sec - prev_rate_per_sec
        # negative rate difference can only be caused by powerups running out,
        #  so we ignore their impact as it's out of the agent control
        rate_diff = max(rate_diff, 0)
        # negative atoms difference can only be caused by buying an upgrade/building,
        #  which isn't inherently bad
        atoms_diff = max(atoms_diff, -100)
        reward = rate_diff + atoms_diff / 500
        if is_waiting:
            reward *= 0.1
        return round(reward, 4)

    def _remember(self, state: FloatVector, action: int, reward: float,
                  next_state: FloatVector):
        """
        Record an n-step transition by accumulating rewards over `self.n_step` steps.
        """
        # Add the latest experience to the recent buffer
        self.recent_experiences.append((state, action, reward, next_state))

        # If buffer has accumulated n steps, process n-step experience
        if len(self.recent_experiences) == self.n_step:
            # Calculate cumulative n-step reward
            cumulative_reward = sum(
                (self.gamma ** i) * exp[2] for i, exp in enumerate(self.recent_experiences)
            )
            # Extract the initial state and action from n steps ago
            initial_state, initial_action, _, _ = self.recent_experiences[0]
            _, _, _, final_next_state = self.recent_experiences[-1]

            # Store n-step transition in memory
            self.memory.append((initial_state, initial_action, cumulative_reward, final_next_state))
            # Remove the oldest experience from the buffer to maintain n steps
            self.recent_experiences.popleft()

    def _act(self, normalized_state: FloatVector, available_actions: FloatVector) -> int:
        "Choose an action based on the current state, and returns its index"
        if len(available_actions) != self.action_size:
            raise ValueError("The available actions vector length must match the model output size")
        act_values = self.model.predict(normalized_state, verbose="0")
        self.progress_monitor.display_bars(
            act_values[0],
            np.where(available_actions == 0)[0].tolist()
        )
        # Exploration: randomly choose an action
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice(np.where(available_actions == 1)[0]))
        # Exploit: select best action
        # Set unavailable actions to -inf to avoid selection
        return int(np.argmax(act_values[0]))

    def _replay(self, batch_size: int) -> float:
        # Sample a batch of experiences from memory
        minibatch = random.sample(self.memory, batch_size)

        # Unpack the batch into separate arrays for easy access
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        # Removing the `1` in (32, 1, 85)
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        # Predict Q-values for next states in a batch (Double Q-Learning)
        next_q_values = self.model.predict(next_states, verbose="0")
        # Standard Q-learning: use the main model for next state predictions
        max_next_q_values = np.amax(next_q_values, axis=1)
        # Compute the target values for the batch
        targets = rewards + (self.gamma ** self.n_step) * max_next_q_values

        # Predict current Q-values for states (only updating specific actions)
        target_f = self.model.predict(states, verbose="0")
        target_f[np.arange(batch_size), actions] = targets

        # Train the model on the updated Q-values (single epoch, batch training)
        history = self.model.fit(states, target_f, epochs=1, verbose="0", batch_size=batch_size)

        # Update epsilon (for exploration-exploitation balance)
        self.epsilon = max(self.epsilon_min, pow(self.epsilon_decay, batch_size / 400))
        return history.history["loss"][0]

    def _replay_old(self, batch_size: int) -> float:
        minibatch = random.sample(self.memory, batch_size)
        for state, action, cumulative_reward, next_state in minibatch:
            target = (
                cumulative_reward
                + (
                    (self.gamma ** self.n_step)
                    * np.amax(self.model.predict(next_state, verbose="0")[0])
                )
            )
            # Update the Q-value for the action taken
            target_f = self.model.predict(state, verbose="0")
            target_f[0][action] = target
            # Train the model on this single sample
            history = self.model.fit(state, target_f, epochs=1, verbose="0")
        self.epsilon = max(self.epsilon_min, pow(self.epsilon_decay, batch_size / 400))
        print(f"new epsilon: {self.epsilon:.4f}")
        return history.history["loss"][0]

def normalize_large_number(number: float) -> float:
    "Normalize a large number (up to 1e22) to a value between -1 and 1"
    if abs(number) < 1:
        return number / 50
    sign = -1 if number < 0 else 1
    number = abs(number)
    return sign * ((number if number == 0 else np.log(number)) + 1) / 50

def unnormalize_large_number(number: float) -> float:
    "Unnormalize a float vector"
    return np.round(np.exp(number * 50), 5)

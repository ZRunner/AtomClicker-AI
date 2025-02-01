import threading
import time
import traceback

from termcolor import cprint
from selenium.common.exceptions import WebDriverException

from src.data_recorder import DataRecorder
from src.dqn_agent import DQNAgent
from src.web import Web

SESSIONS_COUNT = 10
SECONDS_BETWEEN_STEPS = 0.33
MIN_EXPERIMENT_DURATION = 8
MAX_EXPERIMENT_DURATION = 500
SECONDS_BETWEEN_TRAINING = 15


def train_in_background(agent: DQNAgent, data_recorder: DataRecorder):
    "Start a concurrent thread to train the agent"
    def loop():
        "Train the agent in a loop"
        time.sleep(SECONDS_BETWEEN_TRAINING)
        while True:
            if agent.has_stopped:
                break
            # cprint("Starting training", "light_green")
            try:
                if record := agent.train():
                    data_recorder.record_training(record)
            except RuntimeError as err:
                cprint(f"Training failed: {err}", "light_red")
            if agent.has_stopped:
                break
            time.sleep(SECONDS_BETWEEN_TRAINING)

    training_thread = threading.Thread(target=loop)
    training_thread.daemon = True
    training_thread.start()

def run_one_agent():
    "Run the main loop"
    web_client = Web()
    data_recorder = DataRecorder()
    agent = DQNAgent(prod=False)

    previous_memory = data_recorder.get_memory_from_file()
    agent.memory.extend(previous_memory)
    if len(agent.memory) > 10:
        if record := agent.train():
            data_recorder.record_training(record)

    web_client.start_new_browser()
    start = time.time()
    state = web_client.get_state()
    agent.remember(state)
    agent.progress_monitor.update_duration_bar(start, MAX_EXPERIMENT_DURATION)

    train_in_background(agent, data_recorder)

    while True:
        data_recorder.on_action_start()

        action = agent.act(state)
        try:
            web_client.execute_action(action)
        except WebDriverException as err:
            cprint(f"Action failed: {err}\n{traceback.format_exc()}", "red")
            break
        new_state = web_client.get_state()
        reward = agent.remember(new_state)

        data_recorder.on_action_stop(state, action, reward)
        if agent.memory:
            data_recorder.record_memory(agent.memory[-1])

        state = new_state

        now = time.time()
        time_spent = now - start
        if time_spent > MAX_EXPERIMENT_DURATION:
            cprint("Maximal duration reached", "light_red")
            break
        if data_recorder.get_time_since_last_active_action() > MIN_EXPERIMENT_DURATION:
            cprint("Agent is stuck waiting, stopping the experiment", "red")
            break
        if data_recorder.get_time_since_last_active_action() > 30 and state.rate_per_sec < 0.1:
            cprint("Rate is too low, stopping the experiment", "red")
            break
        time.sleep(SECONDS_BETWEEN_STEPS)

    try:
        storage = web_client.extract_local_storage()
    except WebDriverException:
        storage = None

    web_client.quit_browser()
    agent.stop()

    data_recorder.record_final_result(state, len(agent.memory), storage)

    print(
        "Average time per action:", round(data_recorder.get_average_time_per_action(), 3),
        "s for", data_recorder.actions_count, "actions",
        "in", round(time_spent, 3), "seconds"
    )
    print("Final score:", state.atoms_count,
          "atoms with a rate of", state.rate_per_sec, "atoms/s")


if __name__ == "__main__":
    for _ in range(SESSIONS_COUNT):
        cprint(f"### TRAINING SESSION {_ + 1}/{SESSIONS_COUNT} ###", attrs=["bold"])
        try:
            run_one_agent()
            print()
        except Exception as err: # pylint: disable=broad-except
            cprint(f"An error occurred: {err}\n{traceback.format_exc()}", "red")
            time.sleep(1)

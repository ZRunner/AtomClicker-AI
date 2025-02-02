import threading
import time
import traceback
import argparse
from collections import namedtuple

from termcolor import cprint
from selenium.common.exceptions import WebDriverException

from src.data_recorder import DataRecorder
from src.dqn_agent import DQNAgent
from src.web import Web

SECONDS_BETWEEN_STEPS = 0.33
MIN_EXPERIMENT_DURATION = 15

ProgramConfig = namedtuple("Args", ["sessions", "max_duration", "train_every", "headless"])
def parse_args():
    "Parse command line arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sessions", type=int, default=10,
                        help="Number of training sessions")
    parser.add_argument("-d", "--max-duration", type=int, default=500,
                        help="Maximal duration of one session")
    parser.add_argument("-t", "--train-every", type=int, default=15,
                        help="Time between training steps")
    parser.add_argument("-hl", "--headless", action="store_true",
                        help="Run the browser in headless mode")
    parsed = parser.parse_args()
    return ProgramConfig(parsed.sessions, parsed.max_duration, parsed.train_every, parsed.headless)


def train_in_background(agent: DQNAgent, data_recorder: DataRecorder, config: ProgramConfig):
    "Start a concurrent thread to train the agent"
    def loop():
        "Train the agent in a loop"
        time.sleep(config.train_every)
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
            time.sleep(config.train_every)

    training_thread = threading.Thread(target=loop)
    training_thread.daemon = True
    training_thread.start()

def run_one_agent(config: ProgramConfig):
    "Run the main loop"
    web_client = Web()
    data_recorder = DataRecorder()
    agent = DQNAgent(prod=False)

    previous_memory = data_recorder.get_memory_from_file()
    agent.memory.extend(previous_memory)
    if len(agent.memory) > 10:
        if record := agent.train():
            data_recorder.record_training(record)

    web_client.start_new_browser(headless=config.headless)
    start = time.time()
    state = web_client.get_state()
    agent.remember(state)
    agent.progress_monitor.update_duration_bar(start, config.max_duration)

    train_in_background(agent, data_recorder, config)

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
        if time_spent > config.max_duration:
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
    args = parse_args()
    for _ in range(args.sessions):
        cprint(f"### TRAINING SESSION {_ + 1}/{args.sessions} ###", attrs=["bold"])
        try:
            run_one_agent(args)
            print()
        except Exception as err: # pylint: disable=broad-except
            cprint(f"An error occurred: {err}\n{traceback.format_exc()}", "red")
            time.sleep(1)

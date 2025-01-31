import threading
import time
import traceback

from termcolor import cprint
from selenium.common.exceptions import WebDriverException

from src.data_recorder import DataRecorder
from src.dqn_agent import DQNAgent
from src.web import Web

SECONDS_BETWEEN_STEPS = 0.33
MIN_EXPERIMENT_DURATION = 8
SECONDS_BETWEEN_TRAINING = 30

def train_in_background(agent: DQNAgent, data_recorder: DataRecorder):
    "Start a concurrent thread to train the agent"
    def loop():
        "Train the agent in a loop"
        time.sleep(SECONDS_BETWEEN_TRAINING)
        while True:
            if agent.has_stopped:
                break
            cprint("Starting training", "light_green")
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


def main():
    "Run a trained agent for as long as possible."
    web_client = Web()
    data_recorder = DataRecorder(prod=True)
    agent = DQNAgent(prod=True)

    # load both memories from training and prod sessions
    previous_memory = (
        DataRecorder(prod=False).get_memory_from_file()
        + data_recorder.get_memory_from_file()
    )
    agent.memory.extend(previous_memory)
    cprint(f"Loaded {len(agent.memory)} memories from file", "light_blue")
    if len(agent.memory) > 10:
        cprint("Starting training with the loaded memories", "light_green")
        if record := agent.train():
            data_recorder.record_training(record)

    web_client.start_signedin_browser()
    start = time.time()
    state = web_client.get_state()
    agent.remember(state)

    train_in_background(agent, data_recorder)

    while True:
        try:
            data_recorder.on_action_start()

            action = agent.act(state)
            # if action not in {DefaultActions.WAIT, DefaultActions.CLICK_CENTER}:
            #     cprint(f"agent chose action {action}", "cyan")
            web_client.execute_action(action)
            new_state = web_client.get_state()
            reward = agent.remember(new_state)

            data_recorder.on_action_stop(state, action, reward)
            if agent.memory:
                data_recorder.record_memory(agent.memory[-1])

            state = new_state

            if data_recorder.get_time_since_last_active_action() > MIN_EXPERIMENT_DURATION:
                cprint("Agent is stuck waiting, stopping the experiment", "red")
                break
            time.sleep(SECONDS_BETWEEN_STEPS)

        except KeyboardInterrupt:
            cprint("Experiment stopped", "red")
            break
        except WebDriverException:
            cprint("Browser crashed, stopping the experiment", "red")
            break

    try:
        storage = web_client.extract_local_storage()
    except WebDriverException:
        storage = None

    web_client.quit_browser()
    agent.stop()

    data_recorder.record_final_result(state, len(agent.memory), storage)

    time_spent = time.time() - start
    print(
        "Average time per action:", round(data_recorder.get_average_time_per_action(), 3),
        "s for", data_recorder.actions_count, "actions",
        "in", round(time_spent, 3), "seconds"
    )
    print("Final score:", state.atoms_count,
          "atoms with a rate of", state.rate_per_sec, "atoms/s")


if __name__ == "__main__":
    try:
        main()
    except Exception as err: # pylint: disable=broad-except
        cprint("An error occurred, stopping the experiment", "red")
        traceback.print_exc()

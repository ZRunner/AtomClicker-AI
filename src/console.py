import re
import time
from enum import StrEnum
from time import sleep

import numpy as np
from termcolor import cprint


class ProgressMonitor:
    "Use the console to show the training progress."

    def __init__(self, actions_list: list[str]) -> None:
        max_action_length = max(len(action) for action in actions_list)
        self.bars = [
            Bar(1.0, action.ljust(max_action_length), Color.CYAN, show_percentage=False)
            for action in actions_list
        ]
        self.last_training_text = ""
        self.last_action = ""
        self.start_timestamp: float | None = None
        self.time_bar = Bar(1, "Elapsed time:", Color.DARKGREY, show_percentage=False)

    def update_duration_bar(self, start_timestamp: float, max_duration: int):
        "Update the data used to show the duration progress"
        self.start_timestamp = start_timestamp
        self.time_bar.max = max_duration

    def display_bars(self, values: list[float], forbidden_action_indexes: list[float]):
        "Display the bars with the new values."
        # Draw historical data
        cprint(self.last_training_text, "light_green")
        cprint(f"Last meaningful action: {self.last_action}", "light_blue")
        # Draw time bar
        if self.start_timestamp:
            elapsed = time.time() - self.start_timestamp
            self.time_bar.update(elapsed)
        # Draw neural network outputs
        max_index = np.argmax([
            -1 if i in forbidden_action_indexes else value
            for i, value in enumerate(values)
        ])
        max_born = max(1.0, values[max_index])
        for i, action_bar, value in zip(range(len(self.bars)), self.bars, values):
            color = None
            if i in forbidden_action_indexes:
                color = Color.RED
            elif i == max_index:
                color = Color.LIGHTGREEN
            action_bar.max = max_born
            action_bar.update(value, color=color)
        # cleanup old bars
        for _ in range(len(self.bars) + 2):
            _delete_line()
        if self.start_timestamp:
            _delete_line()


class Color(StrEnum):
    "Text colors for the console."
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    ORANGE = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    LIGHTGREY = '\033[37m'
    DARKGREY = '\033[90m'
    LIGHTRED = '\033[91m'
    LIGHTGREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHTBLUE = '\033[94m'
    PINK = '\033[95m'
    LIGHTCYAN = '\033[96m'

    STOP = '\033[0m'

class Bar:
    "A progress bar to display in the console."

    def __init__(self,
                 max_value: float, prefix: str, color: Color = Color.LIGHTGREEN,
                 show_percentage = True) -> None:
        self.max = max_value
        self.prefix = prefix
        self.color = color
        self.show_percentage = show_percentage
        self.width = 70

    def update(self, value: float, color: Color | None = None):
        "Update the bar with the new value."
        progress_normed = min(max(value, 0), self.max) / self.max
        color = color or self.color

        prefix = Color.STOP + self.prefix + ' '
        frac = (
            f"{Color.DARKGREY if value < self.max else Color.GREEN}{value:.3f}"\
            f"/{round(self.max, 3)}"
        )
        if self.show_percentage:
            percent = f"{progress_normed*100:.0f}%".rjust(4)
            suffix = f" {percent} {frac}"
        else:
            suffix = f" {frac}"

        barwidth = self.width - len(_clear_color(prefix))
        barwidth = max(barwidth, 10)

        current_bar = round(min(progress_normed * barwidth, barwidth))
        min_bar = int(min(progress_normed * barwidth, barwidth))

        bar_repr = '|' + color
        if current_bar >= min_bar:
            bar_repr += '━' * current_bar + ' ' * (barwidth - current_bar)
        else:
            bar_repr += '━' * current_bar + '╸' + ' ' * (barwidth - current_bar - 1)
        bar_repr += Color.STOP + '|'
        print(f"{prefix}{bar_repr}{suffix}{Color.STOP}")

def _delete_line():
    "Remove one line from the console."
    print("\033[1A\x1b[2K", end="")

def _clear_color(text):
    text = re.sub(r"\033\[[0-9][0-9]?m", "", text)
    text = re.sub(r"\u001b\[[0-9][0-9]?m", "", text)
    text = re.sub(r"\u001b\[[0-9][0-9]?;1m", "", text)
    return text


if __name__ == "__main__":
    def main():
        "Test the progress bar."
        bar1 = Bar(10, "Progress bar 1")
        bar2 = Bar(10, "Progress bar 2", Color.LIGHTCYAN)

        for i in range(1, 11):
            sleep(1)
            if i % 2 == 0:
                print("i is now", i)
            elif i % 3 == 0:
                print("HMMMMM, i is now", i)
            bar1.update(i)
            bar2.update(i - 0.5)
            if i < 10:
                _delete_line()
                _delete_line()
    main()

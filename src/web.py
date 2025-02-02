import json
import re

from termcolor import cprint
from selenium.common.exceptions import (ElementClickInterceptedException,
                                        NoSuchElementException,
                                        StaleElementReferenceException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver

from .model import (Building, DefaultActions, GameState, LocalStorage, Upgrade,
                   UpgradeType)

UNITS = [
	"",
	"K",
	"M",
	"B",
	"T",
	"Qa",
	"Qi",
	"Sx",
	"Sp",
	"Oc",
	"No",
	"Dc",
	"UDc",
	"DDc",
	"TDc",
	"QaDc",
	"QiDc",
	"SxDc",
	"SpDc",
	"OcDc",
	"NoDc",
	"Vg",
	"UVg",
	"DVg",
	"TVg",
	"QaVg",
	"QiVg",
	"SxVg",
	"SpVg",
	"OcVg",
	"NoVg",
	"Tg",
	"UTg",
	"DTg", # my 2nd run
	"TTg",
	"QaTg",
	"QiTg",
	"SxTg",
	"SpTg", # my 3rd run
	"OcTg",
	"NoTg",
	"Qag",
	"UQag",
	"DQag",
	"TQag", # me
	"QaQag",
	"QiQag",
	"SxQag",
	"SpQag",
	"OcQag",
	"NoQag", # 5
	"Qig",
	"UQig",
	"DQig",
	"TQig",
	"QaQig",
	"QiQig",
	"SxQig",
	"SpQig",
	"OcQig",
	"NoQig",
	"Sxg",
	"USxg",
	"DSxg", # 4
	"TSxg",
	"QaSxg",
	"QiSxg",
	"SxSxg",
	"SpSxg",
	"OcSxg",
	"NoSxg",
	"Spg",
	"USpg",
	"DSpg",
	"TSpg",
	"QaSpg",
	"QiSpg",
	"SxSpg",
	"SpSpg",
	"OcSpg",
	"NoSpg",
	"Ocg",
	"UOcg",
	"DOcg",
	"TOcg", # 3
	"QaOcg",
	"QiOcg",
	"SxOcg",
	"SpOcg",
	"OcOcg",
	"NoOcg",
	"Nog",
	"UNog",
	"DNog",
	"TNog",
	"QaNog", # 2
	"QiNog",
	"SxNog", # 1
	"SpNog",
	"OcNog",
	"NoNog",
	"Dg",
	"UDg",
	"DDg",
	"TDg",
	"QaDg",
	"QiDg",
	"SxDg",
	"SpDg",
	"OcDg",
	"NoDg",
]

NUMERIC_UNIT_REGEX = re.compile(r"^(?P<digits>\d+\.?\d*)(?P<unit>[a-zA-Z]+)?$")
def convert_text_to_float(text: str) -> float:
    "Convert a text to a float"
    match = NUMERIC_UNIT_REGEX.match(text)
    if match is None:
        raise ValueError(f"Invalid text: {text}")
    if not match.group("unit"):
        return float(match.group("digits"))
    string_value, unit = str(match.group("digits")), str(match.group("unit"))
    value_precision = string_value[::-1].index('.')
    numeric_value = int(string_value.replace('.', ''))
    unit_multiplier: int = pow(10, 3 * UNITS.index(unit))
    result = numeric_value * unit_multiplier // (10 ** value_precision)
    return result


class Web:
    "Class to interact with the game through the browser."

    def __init__(self):
        self.driver: WebDriver
        with open("js/extract_game_state.js", encoding="utf8") as file:
            self.extract_game_state_script = file.read()

    def start_new_browser(self, headless = False):
        "Start a new clean browser and open the game."
        cprint("Starting a new browser...", "magenta")
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("https://atom-clicker.ayfri.com/")
        self._wait_ready()

    def start_signedin_browser(self):
        "Start a browser session using previous state and open the game."
        cprint("Booting the previous browser...", "magenta")
        chrome_options = Options()
        chrome_options.add_argument("user-data-dir=selenium")
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("https://atom-clicker.ayfri.com/")
        self._wait_ready()

    def quit_browser(self):
        "Close the browser."
        cprint("Closing the browser...", "magenta")
        self.driver.quit()

    def _find_by_css(self, css_selector: str):
        return self.driver.find_element(by=By.CSS_SELECTOR, value=css_selector)

    def _find_multiple_by_css(self, css_selector: str):
        return self.driver.find_elements(by=By.CSS_SELECTOR, value=css_selector)


    def get_state(self) -> GameState:
        "Return the current game state."
        game_data = self.driver.execute_script(self.extract_game_state_script)
        upgrades = []
        for data in game_data["upgrades"]:
            name = data["name"]
            level = int(name.split()[-1]) if name.split()[-1].isnumeric() else 0
            price = int(convert_text_to_float(data["priceText"]))
            upgrades.append(Upgrade(
                name=name,
                target=self._extract_upgrade_target_from_name(name),
                level=level,
                price=price,
                is_available=data["isAvailable"]
            ))
        buildings = []
        for data in game_data["buildings"]:
            name = data["name"]
            level = int(data["level"] or 0)
            current_rate = convert_text_to_float(data["currentRateText"])
            upgrade_price = int(convert_text_to_float(data["upgradePriceText"]))
            buildings.append(
                Building(name, level, current_rate, upgrade_price, data["isAvailable"])
            )

        return GameState(
            atoms_count=int(convert_text_to_float(game_data["atomsCount"])),
            rate_per_sec=convert_text_to_float(game_data["ratePerSec"]),
            available_powerup=game_data["availablePowerup"],
            buildings=buildings,
            upgrades=upgrades,
        )

    def execute_action(self, action: str):
        "Execute an action."
        if action == DefaultActions.WAIT:
            return
        if action == DefaultActions.CLICK_CENTER:
            self._click_on_atom()
            return
        if action == DefaultActions.CLICK_POWERUP:
            self._click_on_powerup()
            return
        if action.startswith("build_"):
            building_index = int(action.split("_")[1])
            self._click_on_building(building_index)
            return
        if action.startswith("upgrade_"):
            upgrade_index = int(action.split("_")[1])
            self._click_on_upgrade(upgrade_index)
            return
        raise ValueError("Unknown action", action)

    def extract_local_storage(self) -> LocalStorage | None:
        "Extract the local storage of the game."
        json_string = self.driver.execute_script(
            "return window.localStorage.getItem('atomic-clicker-save')"
        )
        return json.loads(json_string) if json_string else None

    def _wait_ready(self):
        "wait for the game to load (we chose to check for the buildings list)"
        WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "buildings"))
        )

    def _extract_upgrade_target_from_name(self, upgrade_name: str) -> UpgradeType:
        upgrade_name = upgrade_name.lower()
        if upgrade_name.split()[-1].isnumeric():
            upgrade_name = " ".join(upgrade_name.split()[:-1])
        names_to_type: dict[str, UpgradeType] = {
            "molecule boost": UpgradeType.BUILDING_MOLECULE,
            "crystal boost": UpgradeType.BUILDING_CRYSTAL,
            "nanostructure boost": UpgradeType.BUILDING_NANOSTRUCTURE,
            "micro-organism boost": UpgradeType.BUILDING_MICROORGANISM,
            "rock boost": UpgradeType.BUILDIN_ROCK,
            "planet boost": UpgradeType.BUILDING_PLANET,
            "star boost": UpgradeType.BUILDING_STAR,
            "neutron star boost": UpgradeType.BUIDLING_NEUTRONSTAR,
            "black hole boost": UpgradeType.BUILDING_BLACKHOLE,
            "click power": UpgradeType.CLICK_POWER_MUL,
            "click value": UpgradeType.CLICK_POWER_VAL,
            "global click power": UpgradeType.CLICK_POWER_APS,
            "global boost": UpgradeType.GLOBAL_BOOST,
            "atom soup": UpgradeType.GLOBAL_ACHIEVEMENTS_MUL,
            "power up interval": UpgradeType.POWERUP_INTERVAL,
            "level boost": UpgradeType.LEVEL_BOOST,
            "unlock levels": UpgradeType.UNLOCK_LEVELS,
        }
        return names_to_type[upgrade_name]

    def _click_on_atom(self):
        "Click on the atom to create more atoms."
        self._find_by_css(".atom").click()

    def _click_on_powerup(self):
        "Click on the powerup to activate it."
        try:
            self._find_by_css(".power-up").click()
        except (
            ElementClickInterceptedException,
            NoSuchElementException,
            StaleElementReferenceException
        ):
            pass

    def _click_on_building(self, building_index: int):
        "Click on a building to upgrade it."
        building = self._find_multiple_by_css(
            ".buildings > div:not([hidden]):has(h3)")[building_index]
        building.click()

    def _click_on_upgrade(self, upgrade_index: int):
        "Click on an upgrade to buy it."
        upgrade = self._find_multiple_by_css(".upgrade")[upgrade_index]
        upgrade.click()

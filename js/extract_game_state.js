// Get the current number of atoms
const atomsCount = document.querySelector('#atoms-value').innerText;

// Get the current rate of created atoms per second
const ratePerSec = document.querySelector('#atoms-per-second-value').innerText;

// Check if a powerup is available
const availablePowerup = document.querySelector('.power-up:not([inert])') !== null;

// Get upgrade data
const upgrades = Array.from(document.querySelectorAll('.upgrade')).slice(0, 10).map(upgrade => {
    const matches = /^(?<name>[\w \d-]+)(?:[^\n]|\n)+Cost: (?<price>\d+\.\d+\w*)\s?$/.exec(upgrade.innerText);
    if (matches === null) {
        return null;
    }
    const { name, price } = matches.groups;
    const isAvailable = !upgrade.classList.contains('cursor-not-allowed');
    return {
        name,
        priceText: price,
        isAvailable
    };
}).filter(upgrade => upgrade !== null);

// Get buildings data
const buildings = Array.from(document.querySelectorAll('.buildings > div:not([hidden]):has(h3)')).map(building => {
    const matches = /^(?<name>[\w-]+)(?: \((?<level>\d+)\))?(?: ⇮\d+)?\n+\D+(?<rate>\d+\.\d+\w*) .+\n+Cost: (?<price>\d+\.\d+\w*)\s?$/i.exec(building.innerText);
    if (matches === null) {
        return null;
    }
    const { name, level, rate, price } = matches.groups;
    const isAvailable = !building.classList.contains('cursor-not-allowed');
    return {
        name,
        level,
        currentRateText: rate,
        upgradePriceText: price,
        isAvailable
    };
}).filter(building => building !== null);

return { atomsCount, ratePerSec, availablePowerup, upgrades, buildings };
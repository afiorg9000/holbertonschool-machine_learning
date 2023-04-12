#!/usr/bin/env python3
""""displays the number of launches per rocket."""

import requests


if __name__ == '__main__':
    response = requests.get("https://api.spacexdata.com/v4/launches").json()

    rockets = {}

    for launch in response:
        rocket = launch['rocket']
        url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket)

        response_rocket = requests.get(url).json()
        rocket_name = response_rocket['name']
        if rocket_name in rockets:
            val = rockets.get(rocket_name)
            rockets[rocket_name] = val + 1
        else:
            rockets[rocket_name] = 1

    for val in sorted(rockets.values(), reverse=True):
        key = [k for k, v in rockets.items() if v == val][0]
        print("{}: {}".format(key, val))

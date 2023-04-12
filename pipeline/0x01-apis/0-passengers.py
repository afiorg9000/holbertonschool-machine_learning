#!/usr/bin/env python3
"""returns the list of ships that can hold a given number of passengers:"""
import requests


def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of"""
    response = requests.get("https://swapi-api.hbtn.io/api/starships/")
    response = response.json()
    ships = []
    next = None

    while(response['next'] is not None):
        if next is not None:
            response = requests.get(next)
            response = response.json()

        for i, ship in enumerate(response['results']):
            try:
                if passengerCount <= int(ship['passengers'].replace(',', '')):
                    ships.append(ship['name'])
            except ValueError:
                if ship['passengers'] != 'n/a' and \
                            ship['passengers'] != 'unknown':
                    ships.append(ship['name'])
        next = response['next']
    return ships

#!/usr/bin/env python3
"""returns the list of ships that can hold a given number of passengers:"""
import requests

def availableShips(passengerCount):
    """returns the list of ships that can hold a given number of passengers:"""
    url = 'https://swapi.dev/api/starships/'
    ships = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            if int(ship['passengers'].replace(',', '')) >= passengerCount:
                ships.append(ship['name'])
        url = data['next']
    return ships

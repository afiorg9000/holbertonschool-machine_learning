#!/usr/bin/env python3
"""returns the list of names of the home planets of all sentient species."""
import requests


def sentientPlanets():
    """returns the list of names of the home planets of all sentient"""
    response = requests.get("https://swapi-api.hbtn.io/api/species/")

    response = response.json()

    planets = []

    next = None

    while(response['next'] is not None):
        if next is not None:
            response = requests.get(next)
            response = response.json()

        for specie in response['results']:
            if specie['classification'] == 'sentient' or \
               specie["designation"] == 'sentient':
                try:
                    response_planet = requests.get(specie['homeworld'])
                    response_planet = response_planet.json()
                    if response_planet['name'] not in planets:
                        planets.append(response_planet['name'])
                except ValueError:
                    pass
        next = response['next']

    return planets

# 0x01 Data Collection - APIs

>

At the end of this project I was able to answer these conceptual questions:

* How to use the Python package `requests`
* How to make HTTP `GET` request
* How to handle rate limit
* How to handle pagination
* How to fetch JSON resources
* How to manipulate data from an external service

## Tasks

0. By using the [Swapi API](https://swapi-api.hbtn.io/), create a method that returns the list of ships that can hold a given number of passengers:

    * Prototype: `def availableShips(passengerCount):`
    * Don’t forget the pagination
    * If no ship available, return an empty list.

1. By using the [Swapi API](https://swapi-api.hbtn.io/), create a method that returns the list of names of the home planets of all `sentient` species.

    * Prototype: `def sentientPlanets():`
    * Don’t forget the pagination

2. By using the [GitHub API](https://docs.github.com/en/rest/users?apiVersion=2022-11-28), write a script that prints the location of a specific user:

    * The user is passed as first argument of the script with the full API URL, example: `./2-user_location.py https://api.github.com/users/holbertonschool`
    * If the user doesn’t exist, print `Not found`
    * If the status code is `403`, print `Reset in X min` where `X` is the number of minutes from now and the value of `X-Ratelimit-Reset`
    * Your code should not be executed when the file is imported (you should use `if __name__ == '__main__':`)

3. By using the [(unofficial) SpaceX API](https://github.com/r-spacex/SpaceX-API/tree/master/docs), write a script that displays the upcoming launch with these information:

    * Name of the launch
    * The date (in local time)
    * The rocket name
    * The name (with the locality) of the launchpad
    Format: `<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)`

    The “upcoming launch” is the one which is the soonest from now, in UTC (we encourage you to use the `date_unix` for sorting it) - and if 2 launches have the same date, use the first one in the API result.

    Your code should not be executed when the file is imported (you should use `if __name__ == '__main__':`)

4. By using the [(unofficial) SpaceX API](https://github.com/r-spacex/SpaceX-API/tree/master/docs), write a script that displays the number of launches per rocket.

    * All launches should be taking in consideration
    * Each line should contain the rocket name and the number of launches separated by `:` (format below in the example)
    * Order the result by the number launches (descending)
    * If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
    * Your code should not be executed when the file is imported (you should use `if __name__ == '__main__':`)


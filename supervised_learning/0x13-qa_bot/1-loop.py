#!/usr/bin/env python3
"""Create a script that takes in input from the user with the prompt"""

while True:
    question = input("Q: ").lower()
    if question in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        exit()
    else:
        print("A:")

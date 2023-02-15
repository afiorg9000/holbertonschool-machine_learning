#!/usr/bin/env python3
"""answers questions from a reference text:"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """answers questions from a reference text:"""
    while True:
        question = input("Q: ").lower()

        if question in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            exit()
        else:
            answer = question_answer(question, reference)
            if answer is None:
                print("A: Sorry, I do not understand your question.")
            else:
                print("A:", answer)

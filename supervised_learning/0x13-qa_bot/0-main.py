#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/ABOUT.md') as f:
    reference = f.read()

print(question_answer('Who is ITG?', reference))
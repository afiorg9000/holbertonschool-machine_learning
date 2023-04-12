#!/usr/bin/env python3
'''40. Top students'''


def top_students(mongo_collection):
    '''A function that returns all
    students sorted by average score'''
    docs = []
    for doc in mongo_collection.find():
        score_sum = sum([s['score'] for s in doc['topics']])
        doc['averageScore'] = score_sum / len(doc['topics'])
        docs.append(doc)
    return (sorted(docs, key=lambda a: a['averageScore'], reverse=True))

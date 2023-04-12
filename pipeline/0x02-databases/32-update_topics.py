#!/usr/bin/env python3
"""
Update the document
"""

from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """Update the document in topics attribute"""
    query = { 'name' : name }
    new_values = {'$set': {'topics' : topics} }
    mongo_collection.update_many(query, new_values)

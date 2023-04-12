#!/usr/bin/env python3
"""
Filter by value
"""

from pymongo import MongoClient


def schools_by_topic(mongo_collection, topic):
    """
    Filter by topic value

    Args:
        mongo_collection is the collection from mongo
        topic is the value in the school by filter

    Return:
        list of schools
    """
    schools = mongo_collection.find({
        'topics': {'$in' : [topic]}})
    return schools

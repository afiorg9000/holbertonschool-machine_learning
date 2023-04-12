#!/usr/bin/env python3
"""
Insert document based on **kwards
"""

from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """Insert Documento in a Collection

    Args:
        mongo_colection is a Collection from mongo

    Return:
        the new _id from new document
    """
    dict = kwargs
    new_id = mongo_collection.insert_one(dict)
    return (new_id.inserted_id)

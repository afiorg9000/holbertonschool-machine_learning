#!/usr/bin/env python3
"""
List of documents with python-mongo
"""

from pymongo import MongoClient


def list_all(mongo_collection):
    """
    List all documents

    Args:
        collection of data
    Return:
        List of documents
    """
    documents = mongo_collection.find({})
    return documents

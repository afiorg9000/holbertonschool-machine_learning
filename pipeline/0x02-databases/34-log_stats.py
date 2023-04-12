#!/usr/bin/env python3
"""
Python script that provides some stats about Nginx logs stored in MongoDB.
"""
import pymongo


if __name__ == "__main__":
    # Connect to the database and collection
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client.logs
    collection = db.nginx

    # Count the number of documents in the collection
    logs_count = collection.count_documents({})

    # Get the count of each HTTP method
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    methods_count = []
    for method in methods:
        method_count = collection.count_documents({"method": method})
        methods_count.append(method_count)

    # Get the count of status checks
    status_check_count = collection.count_documents({"method": "GET", "path": "/status"})

    # Print the results
    print("{} logs".format(logs_count))
    print("Methods:")
    for i in range(len(methods)):
        print("\tmethod {}: {}".format(methods[i], methods_count[i]))
    print("{} status check".format(status_check_count))

#!/usr/bin/env python3
"""
Provide stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017')
    logs = client.logs.nginx
    count_logs = logs.count_documents({})
    print("{} logs".format(count_logs))

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        method_count = logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, method_count))

    status_count = logs.count_documents({"method": "GET", "path": "/status"})
    print("{} status check".format(status_count))

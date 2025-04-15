#!/usr/bin/env python3

import os
import sys
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import numpy as np

def connect_to_cosmos():
    """Connect to the Azure Cosmos DB for MongoDB database."""
    # Connection string should be in environment variable for security
    connection_string = os.environ.get("COSMOS_CONNECTION_STRING")
    
    if not connection_string:
        print("Error: COSMOS_CONNECTION_STRING environment variable not set")
        print("Please set it using: export COSMOS_CONNECTION_STRING='your-connection-string'")
        sys.exit(1)
    
    try:
        # Connect to the MongoDB client
        client = MongoClient(connection_string)
        # Ping to verify connection
        client.admin.command('ping')
        print("Successfully connected to Azure Cosmos DB for MongoDB")
        return client
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_collection_stats(client, db_name, collection_name):
    """Get statistics for the specified collection."""
    try:
        db = client[db_name]
        collection = db[collection_name]
        
        # Get total document count
        doc_count = collection.count_documents({})
        print(f"\nTotal documents in {collection_name}: {doc_count}")
        
        # Get basic statistics
        print("\n=== Collection Statistics ===")
        
        # Check for apps with free tier
        free_tier_count = collection.count_documents({"has_free_tier": True})
        print(f"Apps with free tier: {free_tier_count} ({(free_tier_count/doc_count)*100:.2f}%)")
        
        # Check for apps with free trial
        free_trial_count = collection.count_documents({"has_free_trial": True})
        print(f"Apps with free trial: {free_trial_count} ({(free_trial_count/doc_count)*100:.2f}%)")
        
        # Count by price model type
        print("\n=== Price Model Types ===")
        price_model_pipeline = [
            {"$unwind": "$price_model_type"},
            {"$group": {"_id": "$price_model_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        price_models = list(collection.aggregate(price_model_pipeline))
        for model in price_models:
            print(f"{model['_id']}: {model['count']} ({(model['count']/doc_count)*100:.2f}%)")
        
        # Get pricing tier statistics
        print("\n=== Pricing Tier Statistics ===")
        tiers_pipeline = [
            {"$project": {"app_name": 1, "tier_count": {"$size": {"$ifNull": ["$pricing_tiers", []]}}}},
            {"$group": {
                "_id": "$tier_count",
                "count": {"$sum": 1},
                "apps": {"$push": "$app_name"}
            }},
            {"$sort": {"_id": 1}}
        ]
        tier_stats = list(collection.aggregate(tiers_pipeline))
        for stat in tier_stats:
            print(f"Apps with {stat['_id']} pricing tiers: {stat['count']} ({(stat['count']/doc_count)*100:.2f}%)")
        
        # Get price ranges for paid tiers
        print("\n=== Monthly Price Statistics ===")
        price_pipeline = [
            {"$unwind": {"path": "$pricing_tiers", "preserveNullAndEmptyArrays": False}},
            {"$match": {"pricing_tiers.monthly_price": {"$ne": None, "$gt": 0}}},
            {"$group": {
                "_id": "$app_id",
                "app_name": {"$first": "$app_name"},
                "min_price": {"$min": "$pricing_tiers.monthly_price"},
                "max_price": {"$max": "$pricing_tiers.monthly_price"}
            }},
            {"$group": {
                "_id": None,
                "count": {"$sum": 1},
                "avg_min_price": {"$avg": "$min_price"},
                "avg_max_price": {"$avg": "$max_price"},
                "min_price": {"$min": "$min_price"},
                "max_price": {"$max": "$max_price"}
            }}
        ]
        price_stats = list(collection.aggregate(price_pipeline))
        if price_stats:
            stats = price_stats[0]
            print(f"Apps with paid tiers: {stats['count']}")
            print(f"Average minimum price: ${stats['avg_min_price']:.2f}/month")
            print(f"Average maximum price: ${stats['avg_max_price']:.2f}/month")
            print(f"Lowest monthly price: ${stats['min_price']:.2f}")
            print(f"Highest monthly price: ${stats['max_price']:.2f}")
        
        # Get extraction timestamp statistics
        print("\n=== Extraction Timestamp Statistics ===")
        newest_doc = collection.find_one({}, sort=[("extraction_timestamp", -1)])
        oldest_doc = collection.find_one({}, sort=[("extraction_timestamp", 1)])
        
        if newest_doc and oldest_doc:
            newest_date = newest_doc.get("extraction_timestamp")
            oldest_date = oldest_doc.get("extraction_timestamp")
            if isinstance(newest_date, str):
                newest_date = datetime.fromisoformat(newest_date.replace('Z', '+00:00'))
            if isinstance(oldest_date, str):
                oldest_date = datetime.fromisoformat(oldest_date.replace('Z', '+00:00'))
                
            print(f"Newest extraction: {newest_date}")
            print(f"Oldest extraction: {oldest_date}")
            if isinstance(newest_date, datetime) and isinstance(oldest_date, datetime):
                time_span = newest_date - oldest_date
                print(f"Data spans {time_span.days} days")
        
        # Get sample document schemas
        sample_docs = list(collection.aggregate([{"$sample": {"size": 5}}]))
        fields = set()
        for doc in sample_docs:
            fields.update(doc.keys())
        
        print(f"\n=== Document Schema (from {len(sample_docs)} samples) ===")
        print(f"Fields detected: {len(fields)}")
        print(f"Fields: {', '.join(sorted(fields))}")
        
        return doc_count
        
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return 0

def main():
    """Main function to execute the script."""
    db_name = "apicus"  # Update with your actual database name
    collection_name = "apicus-apps-prices"
    
    # Connect to the database
    client = connect_to_cosmos()
    
    # Get collection statistics
    total_docs = get_collection_stats(client, db_name, collection_name)
    
    # Close the connection
    client.close()
    
    return total_docs

if __name__ == "__main__":
    print("=== Azure Cosmos DB for MongoDB Collection Statistics ===")
    total = main()
    print(f"\nAnalysis complete. Processed {total} documents.") 
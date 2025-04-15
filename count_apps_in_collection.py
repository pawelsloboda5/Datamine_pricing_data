#!/usr/bin/env python3

import os
import sys
from pymongo import MongoClient

def connect_to_cosmos():
    """Connect to the Azure Cosmos DB for MongoDB database."""
    connection_string = os.environ.get("MONGODB_URI") 
    
    if not connection_string:
        print("Error: MONGODB_URI environment variable not set.")
        print("Please set it using: export MONGODB_URI='your-connection-string'")
        sys.exit(1)
    
    try:
        client = MongoClient(connection_string)
        client.admin.command('ping')
        print("Successfully connected to Azure Cosmos DB for MongoDB.")
        return client
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def count_apps_needing_pricing(client, db_name, collection_name):
    """Count apps that have actions/triggers but lack pricing_data."""
    print(f"\nCounting documents in '{db_name}.{collection_name}'...")
    
    count = 0
    try:
        db = client[db_name]
        collection = db[collection_name]
        
        # Define the query criteria using $and and $or
        query = {
            '$and': [
                 { # First condition: Pricing data is missing or null
                     '$or': [
                         { "pricing_data": None },
                         { "pricing_data": { "$exists": False } }
                     ]
                 },
                 { # Second condition: Has actions or triggers
                     '$or': [
                         { "has_actions": True },
                         { "has_triggers": True }
                     ]
                 }
            ]
        }
        
        count = collection.count_documents(query)
        
        print(f"\nFound {count} documents matching the criteria:")
        print(f"  - 'pricing_data' does not exist or is null")
        print(f"  - 'has_actions' is true OR 'has_triggers' is true")
        
    except Exception as e:
        print(f"\nError during count: {e}")
        
    return count

def main():
    """Main function to execute the script."""
    db_name = os.getenv("MONGODB_DB_NAME", "apicus-db-data") # Use env var or default
    collection_name = "apicus-processed-apps" # Target collection
    
    # Connect to the database
    client = connect_to_cosmos()
    
    # Perform the count
    app_count = count_apps_needing_pricing(client, db_name, collection_name)
    
    # Close the connection
    print("\nClosing connection.")
    client.close()
    
    return app_count

if __name__ == "__main__":
    print("=== Counting Apps Needing Pricing Data ===")
    total = main()
    print(f"\nScript finished. Found {total} apps requiring pricing data generation.") 
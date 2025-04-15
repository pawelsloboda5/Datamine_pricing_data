# File: get_apps_with_pricing.py
import os
import sys
import json
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

def find_apps_with_pricing(client, db_name, collection_name):
    """Find all apps in the collection that have the pricing_data field."""
    print(f"\nQuerying collection '{db_name}.{collection_name}' for apps with pricing_data...")
    
    app_names_with_pricing = []
    try:
        db = client[db_name]
        collection = db[collection_name]
        
        # Query for documents where pricing_data exists and is not null
        # Project only the 'name' field
        query = {"pricing_data": {"$exists": True, "$ne": None}}
        projection = {"name": 1, "_id": 0} # Only get the name field
        
        cursor = collection.find(query, projection)
        
        count = 0
        for doc in cursor:
            if "name" in doc and doc["name"]:
                app_names_with_pricing.append(doc["name"])
                count += 1
            else:
                 print("Warning: Found document with pricing_data but missing 'name' field.")

        print(f"Found {count} apps with pricing_data.")
        
    except Exception as e:
        print(f"\nError during query: {e}")
        
    return app_names_with_pricing

def save_to_json(data, filename="apps_with_pricing.json"):
    """Save the list of app names to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved app names to '{filename}'.")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

def main():
    """Main function to execute the script."""
    db_name = "apicus-db-data"  # Verify this is your database name
    collection_name = "apicus-processed-apps" # Target collection
    output_filename = "apps_with_pricing.json"
    
    # Connect to the database
    client = connect_to_cosmos()
    
    # Find apps with pricing data
    app_list = find_apps_with_pricing(client, db_name, collection_name)
    
    # Save the list to JSON
    if app_list:
        save_to_json(app_list, output_filename)
    else:
        print("No apps with pricing_data found to save.")
    
    # Close the connection
    print("\nClosing connection.")
    client.close()

if __name__ == "__main__":
    print("=== Generating List of Apps with Pricing Data ===")
    main()
    print("\nScript finished.")
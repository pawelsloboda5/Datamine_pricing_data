# File: create_pricing_index.py
import os
import sys
from pymongo import MongoClient, ASCENDING

def connect_to_cosmos():
    """Connect to the Azure Cosmos DB for MongoDB database."""
    # Connection string should be in environment variable for security
    connection_string = os.environ.get("MONGODB_URI")
    
    if not connection_string:
        print("Error: MONGODB_URI environment variable not set")
        print("Please set it using: export MONGODB_URI='your-connection-string'")
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

def create_pricing_data_index(client, db_name, collection_name):
    """Create an index on the pricing_data field."""
    try:
        db = client[db_name]
        collection = db[collection_name]
        index_name = "pricing_data_exists_index"
        
        print(f"\nAttempting to create index '{index_name}' on {db_name}.{collection_name}...")
        
        # Check if index already exists
        existing_indexes = collection.index_information()
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists.")
            return

        # Create the index
        result = collection.create_index([("pricing_data", ASCENDING)], name=index_name)
        
        print(f"Successfully created index: {result}")
        
    except Exception as e:
        print(f"Error creating index: {e}")

def main():
    """Main function to execute the script."""
    db_name = "apicus-db-data"  # Verify this is your database name
    collection_name = "apicus-processed-apps" # Target collection
    
    # Connect to the database
    client = connect_to_cosmos()
    
    # Create the index
    create_pricing_data_index(client, db_name, collection_name)
    
    # Close the connection
    print("\nClosing connection.")
    client.close()

if __name__ == "__main__":
    print("=== Creating Pricing Data Existence Index ===")
    main()
    print("\nIndex creation process finished.")

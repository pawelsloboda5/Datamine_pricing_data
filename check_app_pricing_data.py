#!/usr/bin/env python3

import os
import sys
import re
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

def check_apps_for_pricing(client, db_name, collection_name, app_names):
    """Check if specified apps exist in the collection and have pricing_data."""
    print(f"Checking collection '{db_name}.{collection_name}' for {len(app_names)} apps...")
    
    try:
        db = client[db_name]
        collection = db[collection_name]
        
        apps_found = 0
        apps_with_pricing = 0
        apps_missing_pricing = []
        apps_not_found = []

        for app_name in app_names:
            # Use case-insensitive regex for matching app name
            query = {"name": {"$regex": f"^{re.escape(app_name)}$", "$options": "i"}}
            
            # Check if the app exists at all
            app_doc = collection.find_one(query)
            
            if app_doc:
                # App found, now check for pricing_data
                if "pricing_data" in app_doc and app_doc["pricing_data"] is not None:
                    print(f"  [OK] '{app_name}': Found with pricing_data.")
                    apps_found += 1
                    apps_with_pricing += 1
                else:
                    print(f"  [MISSING] '{app_name}': Found, but pricing_data is missing or null.")
                    apps_found += 1
                    apps_missing_pricing.append(app_name)
            else:
                print(f"  [NOT FOUND] '{app_name}': Not found in the collection.")
                apps_not_found.append(app_name)
                
        print("--- Summary ---")
        print(f"Total apps checked: {len(app_names)}")
        print(f"Apps found in collection: {apps_found}")
        print(f"Apps found with pricing_data: {apps_with_pricing}")
        
        if apps_missing_pricing:
            print(f"Apps FOUND but MISSING pricing_data ({len(apps_missing_pricing)}):")
            for app in apps_missing_pricing:
                print(f"  - {app}")
        else:
            print("All found apps have pricing_data.")
            
        if apps_not_found:
            print(f"Apps NOT FOUND in the collection ({len(apps_not_found)}):")
            for app in apps_not_found:
                print(f"  - {app}")
        else:
             print("All specified apps were found in the collection.")

    except Exception as e:
        print(f"Error during check: {e}")

def main():
    """Main function to execute the script."""
    db_name = "apicus"  # Verify this is your database name
    collection_name = "apicus-processed-apps" # Target collection
    
    # List of apps extracted from the template prompts
    apps_to_check = [
        "Shopify", "HubSpot", "Slack", "Facebook Lead Ads", 
        "Salesforce", "Mailchimp", "Calendly", "Google Calendar", 
        "Stripe", "Zendesk", "Gmail", "Trello", "Google Sheet"
    ]
    
    # Connect to the database
    client = connect_to_cosmos()
    
    # Perform the check
    check_apps_for_pricing(client, db_name, collection_name, apps_to_check)
    
    # Close the connection
    print("Closing connection.")
    client.close()

if __name__ == "__main__":
    print("=== Checking Apps for Pricing Data ===")
    main()
    print("Check complete.") 
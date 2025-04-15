"""
Script to merge pricing data from apicus-apps-prices collection
into the corresponding documents in the apicus-processed-apps collection.

This script:
1. Connects to Azure Cosmos DB for MongoDB
2. Creates a backup of the processed apps collection
3. Reads all documents from the apicus-apps-prices collection
4. For each document, extracts the app metadata
5. Finds the corresponding document in apicus-processed-apps collection
6. Adds pricing information to that document
7. Updates the document in apicus-processed-apps collection
8. Generates a CSV report of the operation
"""

import os
import time
import logging
import csv
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB connection settings
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "apicus-db-data")

# Collection names
PRICES_COLLECTION = "apicus-apps-prices"
PROCESSED_APPS_COLLECTION = "apicus-processed-apps"
BACKUP_COLLECTION = f"{PROCESSED_APPS_COLLECTION}-backup-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

# Report file
REPORT_FILENAME = f"pricing_merge_report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"

def connect_to_mongodb():
    """Connect to MongoDB and return database client."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        logger.info(f"Successfully connected to MongoDB database: {MONGODB_DB_NAME}")
        return client, db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def backup_processed_apps_collection(db):
    """Create a backup of the processed apps collection before making changes."""
    try:
        # Get source and backup collections
        source_collection = db[PROCESSED_APPS_COLLECTION]
        backup_collection = db[BACKUP_COLLECTION]
        
        # Get all documents from source collection
        source_docs = list(source_collection.find({}))
        
        if not source_docs:
            logger.warning(f"Source collection {PROCESSED_APPS_COLLECTION} is empty, no backup created")
            return
        
        # Insert all documents into backup collection
        backup_collection.insert_many(source_docs)
        
        logger.info(f"Created backup of {len(source_docs)} documents from {PROCESSED_APPS_COLLECTION} to {BACKUP_COLLECTION}")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise

def get_pricing_documents(db):
    """Get all documents from the pricing collection."""
    try:
        prices_collection = db[PRICES_COLLECTION]
        pricing_docs = list(prices_collection.find({}))
        logger.info(f"Retrieved {len(pricing_docs)} documents from {PRICES_COLLECTION} collection")
        return pricing_docs
    except Exception as e:
        logger.error(f"Error retrieving pricing documents: {e}")
        raise

def update_processed_app_document(db, pricing_doc, dry_run=False):
    """Update the corresponding document in processed apps collection with pricing data."""
    try:
        # Get the app metadata from the pricing document
        app_metadata = pricing_doc.get("original_app_metadata", {})
        app_id = app_metadata.get("app_id")
        slug = app_metadata.get("slug")
        name = app_metadata.get("name")
        
        if not any([app_id, slug, name]):
            logger.warning(f"Missing identification data in pricing document {pricing_doc.get('_id')}")
            return False, "Missing identification data", None
        
        # Get processed apps collection
        processed_apps_collection = db[PROCESSED_APPS_COLLECTION]
        app_doc = None
        match_method = None
        
        # Try different strategies to find the matching document
        # Strategy 1: Match by app_id (most precise)
        if app_id:
            app_doc = processed_apps_collection.find_one({"app_id": app_id})
            if app_doc:
                match_method = "app_id"
                logger.debug(f"Found match by app_id: {app_id}")
        
        # Strategy 2: Match by slug if app_id match failed
        if not app_doc and slug:
            app_doc = processed_apps_collection.find_one({"slug": slug})
            if app_doc:
                match_method = "slug"
                logger.debug(f"Found match by slug: {slug}")
        
        # Strategy 3: Match by name if app_id and slug matches failed
        if not app_doc and name:
            app_doc = processed_apps_collection.find_one({"name": name})
            if app_doc:
                match_method = "name"
                logger.debug(f"Found match by name: {name}")
        
        if not app_doc:
            error_msg = f"No matching document found for app_id: {app_id}, slug: {slug}, name: {name}"
            logger.warning(error_msg)
            return False, error_msg, None
            
        # Check if pricing_data already exists in the target document
        if "pricing_data" in app_doc and app_doc["pricing_data"] is not None:
            app_display_name = app_metadata.get('name', slug or app_id)
            logger.info(f"Skipping app: {app_display_name} - pricing_data already exists.")
            return True, f"Skipped - pricing data already exists (Matched by {match_method})", app_doc["_id"]
            
        # Extract pricing data (excluding certain fields)
        pricing_data = {k: v for k, v in pricing_doc.items() 
                       if k not in ['_id', 'original_app_metadata', 'extraction_timestamp']}
        
        # Add timestamp for when pricing was added
        pricing_data['pricing_data_added'] = datetime.utcnow()
        
        # Update the app document with pricing data (unless in dry run mode)
        if not dry_run:
            processed_apps_collection.update_one(
                {"_id": app_doc["_id"]},
                {"$set": {"pricing_data": pricing_data}}
            )
            log_action = "Updated"
        else:
            log_action = "Would update (dry run)"
        
        app_display_name = app_metadata.get('name', slug or app_id)
        logger.info(f"{log_action} document for app: {app_display_name}")
        return True, f"Matched by {match_method}", app_doc["_id"]
    
    except Exception as e:
        error_msg = f"Error updating document: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, None

def generate_report(results):
    """Generate a CSV report of the merge operation."""
    try:
        with open(REPORT_FILENAME, 'w', newline='') as csvfile:
            fieldnames = ['app_id', 'name', 'slug', 'success', 'match_method', 'document_id', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
                
        logger.info(f"Report generated: {REPORT_FILENAME}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")

def main(create_backup=True, dry_run=False):
    """Main function to merge pricing data into processed apps."""
    start_time = time.time()
    results = []
    
    try:
        # Connect to MongoDB
        client, db = connect_to_mongodb()
        
        # Create backup if requested and not in dry run mode
        if create_backup and not dry_run:
            backup_processed_apps_collection(db)
        elif create_backup and dry_run:
            logger.info("Skipping backup creation in dry run mode")
        
        # Get pricing documents
        pricing_docs = get_pricing_documents(db)
        
        # Track statistics
        total_docs = len(pricing_docs)
        updated_docs = 0
        failed_docs = 0
        
        # Process each pricing document
        for i, pricing_doc in enumerate(pricing_docs, 1):
            logger.info(f"Processing document {i}/{total_docs}")
            
            # Extract app metadata for reporting
            app_metadata = pricing_doc.get("original_app_metadata", {})
            app_id = app_metadata.get("app_id")
            name = app_metadata.get("name")
            slug = app_metadata.get("slug")
            
            # Update the corresponding processed app document
            success, message, doc_id = update_processed_app_document(db, pricing_doc, dry_run)
            
            # Record result
            result = {
                'app_id': app_id,
                'name': name,
                'slug': slug,
                'success': success,
                'document_id': str(doc_id) if doc_id else None,
                'error': None if success else message,
                'match_method': message if success else None
            }
            results.append(result)
            
            if success:
                updated_docs += 1
            else:
                failed_docs += 1
                
            # Log progress every 100 documents
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{total_docs} documents processed")
        
        # Generate report
        report_prefix = "dry_run_" if dry_run else ""
        global REPORT_FILENAME
        REPORT_FILENAME = f"{report_prefix}{REPORT_FILENAME}"
        generate_report(results)
        
        # Log final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"Completed processing {total_docs} documents in {elapsed_time:.2f} seconds")
        if dry_run:
            logger.info(f"Would have updated: {updated_docs} documents (dry run)")
        else:
            logger.info(f"Successfully updated: {updated_docs} documents")
        logger.info(f"Failed to update: {failed_docs} documents")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
    
    finally:
        # Close MongoDB connection if it was established
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    main() 
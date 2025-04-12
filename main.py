#!/usr/bin/env python3
"""
Main script for the Apicus app pricing scraper and analysis pipeline.
This script coordinates the discovery, extraction, and storage of pricing data.
"""

import argparse
import sys
import time
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import traceback

# Import our modules
import scrape_w_jina_ai
import extract_pricing_data
import bing_grounding_search
import pricing_schema

# Import the pricing_models module
import pricing_models

# Import the pricing_schema module
import pricing_schema


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apicus app pricing scraper and analyzer")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Discovery command
    discover_parser = subparsers.add_parser("discover", help="Discover pricing pages for apps")
    discover_parser.add_argument("--limit", type=int, default=10, 
                               help="Maximum number of apps to process (default: 10)")
    discover_parser.add_argument("--app-id", type=str, 
                               help="Process a specific app by ID")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract pricing data from discovered pages")
    extract_parser.add_argument("--limit", type=int, default=5, 
                              help="Maximum number of apps to process (default: 5)")
    extract_parser.add_argument("--app-id", type=str, 
                              help="Process a specific app by ID")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline (discover + extract)")
    pipeline_parser.add_argument("--discover-limit", type=int, default=10, 
                               help="Maximum number of apps to discover (default: 10)")
    pipeline_parser.add_argument("--extract-limit", type=int, default=5, 
                               help="Maximum number of apps to extract (default: 5)")
    
    # Enhanced pipeline command (new)
    enhanced_parser = subparsers.add_parser("enhanced-pipeline", 
                                          help="Run enhanced pipeline with Bing Search and validation")
    enhanced_parser.add_argument("--limit", type=int, default=5, 
                               help="Maximum number of apps to process (default: 5)")
    enhanced_parser.add_argument("--app-name", type=str,
                               help="Process a specific app by name")
    enhanced_parser.add_argument("--app-slug", type=str,
                               help="Optional URL-friendly version of app name (used with --app-name)")
    enhanced_parser.add_argument("--output-dir", type=str, default="pricing_results",
                               help="Directory to save results (default: pricing_results)")
    
    # Bing Search command
    bing_parser = subparsers.add_parser("bing", help="Search for pricing using Azure Bing Search Grounding")
    bing_parser.add_argument("--app-name", type=str, required=True,
                            help="Name of the app to search for")
    bing_parser.add_argument("--app-slug", type=str,
                            help="Optional URL-friendly version of app name")
    bing_parser.add_argument("--output", type=str, 
                            help="Optional filename to save results as JSON")
    
    # MongoDB Bulk Processing command (new)
    mongodb_parser = subparsers.add_parser("process-mongodb", 
                                         help="Process apps directly from MongoDB and save results")
    mongodb_parser.add_argument("--limit", type=int, default=10,
                              help="Maximum number of apps to process (default: 10)")
    mongodb_parser.add_argument("--output-dir", type=str, default="pricing_results",
                              help="Directory to save intermediate results (default: pricing_results)")
    mongodb_parser.add_argument("--skip", type=int, default=0,
                              help="Number of documents to skip (default: 0)")
    
    # Index command (for future use - will create vector indexes)
    index_parser = subparsers.add_parser("index", help="Create vector indexes for pricing data")
    
    return parser.parse_args()

def run_discovery(args):
    """Run the discovery phase of the pipeline."""
    print("Starting pricing page discovery...")
    
    if args.app_id:
        # TODO: Implement single app processing
        print(f"Processing specific app: {args.app_id}")
        # This would require a modification to the scrape_w_jina_ai module
        # to support processing a single app by ID
        print("Single app processing not yet implemented")
        return
    
    # Run discovery for multiple apps
    scrape_w_jina_ai.process_apps(limit=args.limit)
    
    print(f"Discovery phase completed for up to {args.limit} apps")

def run_extraction(args):
    """Run the extraction phase of the pipeline."""
    print("Starting pricing data extraction...")
    
    if args.app_id:
        # TODO: Implement single app processing
        print(f"Processing specific app: {args.app_id}")
        # This would require a modification to the extract_pricing_data module
        # to support processing a single app by ID
        print("Single app processing not yet implemented")
        return
    
    # Run extraction for multiple apps
    extract_pricing_data.process_pricing_discovery_results(limit=args.limit)
    
    print(f"Extraction phase completed for up to {args.limit} apps")

def run_full_pipeline(args):
    """Run the full pipeline (discovery + extraction)."""
    print("Starting full pricing data pipeline...")
    
    # Run discovery
    print(f"Phase 1: Discovery (limit: {args.discover_limit} apps)")
    scrape_w_jina_ai.process_apps(limit=args.discover_limit)
    
    # Small delay between phases
    time.sleep(2)
    
    # Run extraction
    print(f"Phase 2: Extraction (limit: {args.extract_limit} apps)")
    extract_pricing_data.process_pricing_discovery_results(limit=args.extract_limit)
    
    print("Full pipeline completed successfully")

def run_enhanced_pipeline(args):
    """
    Run enhanced pipeline with all components:
    1. URL discovery (scrape_w_jina_ai.py)
    2. Bing Search grounding (bing_grounding_search.py)
    3. Schema validation (pricing_schema.py)
    4. Pricing extraction (extract_pricing_data.py)
    """
    print("Starting enhanced pricing data pipeline...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If specific app is provided
    if args.app_name:
        print(f"Processing specific app: {args.app_name}")
        process_single_app_enhanced(args.app_name, args.app_slug, args.output_dir)
        return
    
    # Connect to MongoDB
    db = scrape_w_jina_ai.connect_to_mongodb()
    
    # Phase 1: Get apps to process
    print(f"Phase 1: Getting apps to process (limit: {args.limit})")
    apps = scrape_w_jina_ai.get_apps_from_mongodb(db, limit=args.limit)
    print(f"Retrieved {len(apps)} apps for processing")
    
    # Process each app through the pipeline
    for i, app in enumerate(apps):
        app_name = app["name"]
        app_slug = app.get("slug")
        app_id = app["app_id"]
        
        print(f"\n[{i+1}/{len(apps)}] Processing app: {app_name} ({app_slug or 'no slug'})")
        
        # Create app directory
        app_dir = os.path.join(args.output_dir, app_id)
        os.makedirs(app_dir, exist_ok=True)
        
        try:
            # Phase 2: Bing Search grounding
            print(f"Phase 2: Running Bing Search grounding for {app_name}")
            bing_result = bing_grounding_search.main(app_name, app_slug)
            
            # Save Bing results
            bing_result_file = os.path.join(app_dir, "bing_search_results.json")
            with open(bing_result_file, "w", encoding="utf-8") as f:
                json.dump(bing_result, f, indent=2)
            
            print(f"Saved Bing search results to {bing_result_file}")
            
            # Check if we got content extraction
            if "extracted_content" not in bing_result or not bing_result["extracted_content"]:
                print(f"No valid content extracted for {app_name}, skipping to next app")
                continue
            
            # Phase 3: Extract and validate pricing data
            print(f"Phase 3: Extracting and validating pricing data for {app_name}")
            
            best_pricing_data = None
            best_pricing_confidence = 0
            
            # Process each extracted content
            for content_item in bing_result["extracted_content"]:
                url = content_item.get("url", "Unknown URL")
                is_accessible = content_item.get("is_accessible", False)
                content = content_item.get("content")
                
                if not content or not is_accessible:
                    print(f"Skipping inaccessible content from {url}")
                    continue
                
                # Check if pricing is public
                is_pricing_public = extract_pricing_data.check_pricing_is_public(content)
                
                # Extract pricing data using Azure OpenAI
                print(f"Analyzing content from {url}")
                pricing_data = bing_grounding_search.analyze_pricing_with_openai(
                    content, app_name, app_slug, is_pricing_public, is_accessible
                )
                
                if pricing_data:
                    # Add source URL and extraction metadata
                    pricing_data["source_url"] = url
                    pricing_data["extraction_timestamp"] = datetime.now().isoformat()
                    
                    # Validate against schema
                    is_valid = validate_pricing_data(pricing_data)
                    pricing_data["schema_validated"] = is_valid
                    
                    # Calculate confidence score based on completeness
                    confidence = calculate_pricing_confidence(pricing_data)
                    pricing_data["confidence_score"] = confidence
                    
                    # Save individual pricing data
                    url_filename = url.split('/')[-1].split('?')[0][:30]
                    if not url_filename:
                        url_filename = f"url_{hash(url) % 10000}"
                    
                    pricing_file = os.path.join(app_dir, f"pricing_data_{url_filename}.json")
                    with open(pricing_file, "w", encoding="utf-8") as f:
                        json.dump(pricing_data, f, indent=2)
                    
                    print(f"Saved pricing data from {url} with confidence {confidence}%")
                    
                    # Track best pricing data based on confidence
                    if confidence > best_pricing_confidence:
                        best_pricing_data = pricing_data
                        best_pricing_confidence = confidence
            
            # Phase 4: Save final pricing data
            if best_pricing_data:
                # Normalize pricing data before saving
                best_pricing_data = normalize_pricing_data(best_pricing_data)
                
                # Save best pricing data
                final_pricing_file = os.path.join(app_dir, "final_pricing_data.json")
                with open(final_pricing_file, "w", encoding="utf-8") as f:
                    json.dump(best_pricing_data, f, indent=2)
                
                print(f"Saved final pricing data with confidence {best_pricing_confidence}%")
                
                # Store in MongoDB
                try:
                    db = extract_pricing_data.connect_to_mongodb()
                    pricing_collection = db["apicus-apps-prices"]
                    
                    # Check if we already have pricing data for this app
                    existing = pricing_collection.find_one({"app_id": app_id})
                    if existing:
                        # Update if new data has higher confidence
                        if "confidence_score" not in existing or best_pricing_confidence > existing.get("confidence_score", 0):
                            pricing_collection.replace_one({"app_id": app_id}, best_pricing_data)
                            print(f"Updated pricing data in MongoDB with higher confidence ({best_pricing_confidence}%)")
                        else:
                            print(f"Keeping existing pricing data with higher confidence ({existing.get('confidence_score', 0)}%)")
                    else:
                        # Insert new data
                        pricing_collection.insert_one(best_pricing_data)
                        print(f"Inserted new pricing data into MongoDB")
                except Exception as e:
                    print(f"Error storing pricing data in MongoDB: {e}")
            else:
                print(f"No valid pricing data found for {app_name}")
            
        except Exception as e:
            print(f"Error processing {app_name}: {e}")
        
        # Add a small delay between apps
        if i < len(apps) - 1:
            time.sleep(2)
    
    print("\nEnhanced pipeline completed successfully")

def process_single_app_enhanced(app_name, app_slug, output_dir):
    """Process a single app through the enhanced pipeline."""
    # Create app directory
    app_id = app_slug.lower() if app_slug else app_name.lower().replace(" ", "_")
    app_dir = os.path.join(output_dir, app_id)
    os.makedirs(app_dir, exist_ok=True)
    
    try:
        # Bing Search grounding
        print(f"Running Bing Search grounding for {app_name}")
        bing_result = bing_grounding_search.main(app_name, app_slug)
        
        # Save Bing results
        bing_result_file = os.path.join(app_dir, "bing_search_results.json")
        with open(bing_result_file, "w", encoding="utf-8") as f:
            json.dump(bing_result, f, indent=2)
        
        # Extract and validate pricing data
        if "extracted_content" in bing_result and bing_result["extracted_content"]:
            print(f"Extracting and validating pricing data for {app_name}")
            
            best_pricing_data = None
            best_pricing_confidence = 0
            
            for content_item in bing_result["extracted_content"]:
                url = content_item.get("url", "Unknown URL")
                is_accessible = content_item.get("is_accessible", False)
                content = content_item.get("content")
                
                if not content or not is_accessible:
                    continue
                
                # Check if pricing data is already analyzed
                if "pricing_data" in content_item and content_item["pricing_data"]:
                    pricing_data = content_item["pricing_data"]
                    print(f"Using pre-analyzed pricing data from {url}")
                else:
                    # Check if pricing is public
                    is_pricing_public = extract_pricing_data.check_pricing_is_public(content)
                    
                    # Extract pricing data
                    pricing_data = bing_grounding_search.analyze_pricing_with_openai(
                        content, app_name, app_slug, is_pricing_public, is_accessible
                    )
                
                if pricing_data:
                    # Add source URL and metadata
                    pricing_data["source_url"] = url
                    pricing_data["extraction_timestamp"] = datetime.now().isoformat()
                    
                    # Validate against schema
                    is_valid = validate_pricing_data(pricing_data)
                    pricing_data["schema_validated"] = is_valid
                    
                    # Calculate confidence
                    confidence = calculate_pricing_confidence(pricing_data)
                    pricing_data["confidence_score"] = confidence
                    
                    # Save individual pricing data
                    url_filename = url.split('/')[-1].split('?')[0][:30]
                    if not url_filename:
                        url_filename = f"url_{hash(url) % 10000}"
                    
                    pricing_file = os.path.join(app_dir, f"pricing_data_{url_filename}.json")
                    with open(pricing_file, "w", encoding="utf-8") as f:
                        json.dump(pricing_data, f, indent=2)
                    
                    # Track best pricing data
                    if confidence > best_pricing_confidence:
                        best_pricing_data = pricing_data
                        best_pricing_confidence = confidence
            
            # Save final pricing data
            if best_pricing_data:
                # Normalize pricing data before saving
                best_pricing_data = normalize_pricing_data(best_pricing_data)
                
                final_pricing_file = os.path.join(app_dir, "final_pricing_data.json")
                with open(final_pricing_file, "w", encoding="utf-8") as f:
                    json.dump(best_pricing_data, f, indent=2)
                
                print(f"Saved final pricing data with confidence {best_pricing_confidence}%")
                
                # Store in MongoDB if available
                try:
                    db = extract_pricing_data.connect_to_mongodb()
                    if db:
                        pricing_collection = db["apicus-apps-prices"]
                        
                        # Ensure app_id is present
                        if "app_id" not in best_pricing_data:
                            best_pricing_data["app_id"] = app_id
                        
                        existing = pricing_collection.find_one({"app_id": app_id})
                        if existing:
                            if best_pricing_confidence > existing.get("confidence_score", 0):
                                pricing_collection.replace_one({"app_id": app_id}, best_pricing_data)
                                print("Updated pricing data in MongoDB")
                        else:
                            pricing_collection.insert_one(best_pricing_data)
                            print("Inserted pricing data into MongoDB")
                except Exception as e:
                    print(f"Error storing in MongoDB: {e}")
                    traceback.print_exc()
            else:
                print("No valid pricing data found")
        else:
            print("No content extracted from Bing search")
    
    except Exception as e:
        print(f"Error processing {app_name}: {e}")
        traceback.print_exc()

def run_mongodb_processing(args):
    """
    Process apps directly from MongoDB collection apicus-processed-apps,
    filtering for apps that have at least one action or trigger.
    
    Args:
        args: Command line arguments
    """
    print("Starting MongoDB app processing pipeline...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Connect to MongoDB
    db = scrape_w_jina_ai.connect_to_mongodb()
    
    # Query apps with at least one action OR trigger
    query = {
        "$or": [
            {"has_actions": True},
            {"has_triggers": True}
        ]
    }
    
    print(f"Retrieving apps from MongoDB (limit: {args.limit}, skip: {args.skip})...")
    apps_collection = db["apicus-processed-apps"]
    pricing_collection = db["apicus-apps-prices"]
    
    # Find apps with actions/triggers, skipping already processed ones if needed
    apps_cursor = apps_collection.find(query).skip(args.skip).limit(args.limit)
    apps = list(apps_cursor)
    
    print(f"Retrieved {len(apps)} apps for processing")
    
    # Process each app
    for i, app in enumerate(apps):
        app_name = app["name"]
        app_slug = app.get("slug")
        app_id = app["app_id"]
        
        # Check if app has actions or triggers
        has_actions = app.get("has_actions", False)
        has_triggers = app.get("has_triggers", False)
        
        if not has_actions and not has_triggers:
            print(f"Skipping {app_name} - no actions or triggers")
            continue
        
        print(f"\n[{i+1}/{len(apps)}] Processing app: {app_name} ({app_slug or 'no slug'})")
        print(f"  Actions: {has_actions}, Triggers: {has_triggers}")
        
        # Create app directory for intermediate results
        app_dir = os.path.join(args.output_dir, app_id)
        os.makedirs(app_dir, exist_ok=True)
        
        try:
            # Run Bing Search grounding
            print(f"Running Bing Search grounding for {app_name}")
            bing_result = bing_grounding_search.main(app_name, app_slug)
            
            # Save intermediate Bing results
            bing_result_file = os.path.join(app_dir, "bing_search_results.json")
            with open(bing_result_file, "w", encoding="utf-8") as f:
                json.dump(bing_result, f, indent=2)
            
            # Check if we got content extraction
            if "extracted_content" not in bing_result or not bing_result["extracted_content"]:
                print(f"No valid content extracted for {app_name}, skipping to next app")
                continue
            
            # Extract and validate pricing data
            print(f"Extracting and validating pricing data for {app_name}")
            
            best_pricing_data = None
            best_pricing_confidence = 0
            
            # Process each extracted content
            for content_item in bing_result["extracted_content"]:
                url = content_item.get("url", "Unknown URL")
                is_accessible = content_item.get("is_accessible", False)
                content = content_item.get("content")
                
                if not content or not is_accessible:
                    print(f"Skipping inaccessible content from {url}")
                    continue
                
                # Check if pricing is public
                is_pricing_public = extract_pricing_data.check_pricing_is_public(content)
                
                # Extract pricing data using Azure OpenAI
                print(f"Analyzing content from {url}")
                pricing_data = bing_grounding_search.analyze_pricing_with_openai(
                    content, app_name, app_slug, is_pricing_public, is_accessible
                )
                
                if pricing_data:
                    # Add source URL and extraction metadata
                    pricing_data["source_url"] = url
                    pricing_data["extraction_timestamp"] = datetime.now().isoformat()
                    
                    # Add metadata from the original app document
                    pricing_data["original_app_metadata"] = {
                        "app_id": app.get("app_id"),
                        "name": app.get("name"),
                        "slug": app.get("slug"),
                        "description": app.get("description"),
                        "logo_url": app.get("logo_url"),
                        "categories": app.get("normalized_categories", []),
                        "category_slugs": app.get("category_slugs", []),
                        "has_actions": app.get("has_actions", False),
                        "has_triggers": app.get("has_triggers", False),
                        "action_count": app.get("action_count", 0),
                        "trigger_count": app.get("trigger_count", 0)
                    }
                    
                    # Validate against schema
                    is_valid = validate_pricing_data(pricing_data)
                    pricing_data["schema_validated"] = is_valid
                    
                    # Calculate confidence score
                    confidence = calculate_pricing_confidence(pricing_data)
                    pricing_data["confidence_score"] = confidence
                    
                    # Save individual pricing data
                    url_filename = url.split('/')[-1].split('?')[0][:30]
                    if not url_filename:
                        url_filename = f"url_{hash(url) % 10000}"
                    
                    pricing_file = os.path.join(app_dir, f"pricing_data_{url_filename}.json")
                    with open(pricing_file, "w", encoding="utf-8") as f:
                        json.dump(pricing_data, f, indent=2)
                    
                    print(f"Saved pricing data from {url} with confidence {confidence}%")
                    
                    # Track best pricing data based on confidence
                    if confidence > best_pricing_confidence:
                        best_pricing_data = pricing_data
                        best_pricing_confidence = confidence
            
            # Save final pricing data to MongoDB
            if best_pricing_data:
                # Normalize pricing data before saving
                best_pricing_data = normalize_pricing_data(best_pricing_data)
                
                # Save best pricing data locally as well
                final_pricing_file = os.path.join(app_dir, "final_pricing_data.json")
                with open(final_pricing_file, "w", encoding="utf-8") as f:
                    json.dump(best_pricing_data, f, indent=2)
                
                print(f"Saved final pricing data with confidence {best_pricing_confidence}%")
                
                # Store in MongoDB
                try:
                    # Check if we already have pricing data for this app
                    existing = pricing_collection.find_one({"app_id": app_id})
                    if existing:
                        # Update if new data has higher confidence
                        if "confidence_score" not in existing or best_pricing_confidence > existing.get("confidence_score", 0):
                            pricing_collection.replace_one({"app_id": app_id}, best_pricing_data)
                            print(f"Updated pricing data in MongoDB with higher confidence ({best_pricing_confidence}%)")
                        else:
                            print(f"Keeping existing pricing data with higher confidence ({existing.get('confidence_score', 0)}%)")
                    else:
                        # Insert new data
                        pricing_collection.insert_one(best_pricing_data)
                        print(f"Inserted new pricing data into MongoDB")
                except Exception as e:
                    print(f"Error storing pricing data in MongoDB: {e}")
            else:
                print(f"No valid pricing data found for {app_name}")
        
        except Exception as e:
            print(f"Error processing {app_name}: {e}")
            
        # Add a delay between apps to respect API rate limits
        if i < len(apps) - 1:
            delay = 2  # seconds
            print(f"Waiting {delay} seconds before processing next app...")
            time.sleep(delay)
    
    print(f"\nFinished processing {len(apps)} apps from MongoDB")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

def validate_pricing_data(pricing_data):
    """
    Validate pricing data against the schema.
    
    Args:
        pricing_data (dict): Pricing data to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Ensure required fields are present
    required_fields = [
        "app_id", "app_name", "price_model_type", "has_free_tier", 
        "has_free_trial", "currency", "is_pricing_public"
    ]
    
    for field in required_fields:
        if field not in pricing_data:
            print(f"Validation failed: Missing required field '{field}'")
            return False
    
    # Validate data types
    if not isinstance(pricing_data.get("price_model_type"), list):
        print("Validation failed: price_model_type must be a list")
        return False
    
    if not isinstance(pricing_data.get("has_free_tier"), bool):
        print("Validation failed: has_free_tier must be a boolean")
        return False
    
    if not isinstance(pricing_data.get("has_free_trial"), bool):
        print("Validation failed: has_free_trial must be a boolean")
        return False
    
    # If pricing_tiers exists and isn't None, ensure it's an array
    if "pricing_tiers" in pricing_data and pricing_data["pricing_tiers"] is not None and not isinstance(pricing_data["pricing_tiers"], list):
        print("Validation failed: pricing_tiers must be a list")
        return False
    
    # If usage_based_pricing exists and isn't None, ensure it's an array
    if "usage_based_pricing" in pricing_data and pricing_data["usage_based_pricing"] is not None and not isinstance(pricing_data["usage_based_pricing"], list):
        print("Validation failed: usage_based_pricing must be a list")
        return False
    
    # If promotional_offers exists and isn't None, ensure it's an array
    if "promotional_offers" in pricing_data and pricing_data["promotional_offers"] is not None and not isinstance(pricing_data["promotional_offers"], list):
        print("Validation failed: promotional_offers must be a list")
        return False
    
    # If additional_fees exists and isn't None, ensure it's an array
    if "additional_fees" in pricing_data and pricing_data["additional_fees"] is not None and not isinstance(pricing_data["additional_fees"], list):
        print("Validation failed: additional_fees must be a list")
        return False
    
    return True

def normalize_pricing_data(pricing_data):
    """
    Normalize pricing data to ensure it conforms to the expected schema.
    Converts null values to empty arrays and fixes enum formats.
    
    Args:
        pricing_data (dict): Pricing data to normalize
    
    Returns:
        dict: Normalized pricing data
    """
    # Make a copy to avoid modifying the original
    normalized_data = pricing_data.copy()
    
    # Ensure array fields are never null but empty lists
    array_fields = ["pricing_tiers", "usage_based_pricing", "promotional_offers", "additional_fees", "all_pricing_urls"]
    for field in array_fields:
        if field in normalized_data and normalized_data[field] is None:
            normalized_data[field] = []
    
    # Fix enum value formatting in price_model_type (remove PriceModelType. prefix)
    if "price_model_type" in normalized_data and normalized_data["price_model_type"]:
        fixed_types = []
        for model_type in normalized_data["price_model_type"]:
            # Check if the value includes the enum class name
            if isinstance(model_type, str) and model_type.startswith("PriceModelType."):
                # Extract just the value part (after the dot)
                value = model_type.split('.', 1)[1].lower()
                fixed_types.append(value)
            else:
                # Keep the value as is if it's already formatted correctly
                fixed_types.append(model_type)
        normalized_data["price_model_type"] = fixed_types
    
    # Ensure ai_specific_pricing is a dict if present and not None
    if "ai_specific_pricing" in normalized_data and normalized_data["ai_specific_pricing"] is None:
        normalized_data["ai_specific_pricing"] = {}
    
    return normalized_data

def calculate_pricing_confidence(pricing_data):
    """
    Calculate a confidence score (0-100) for the pricing data based on completeness.
    
    Args:
        pricing_data (dict): Pricing data to evaluate
    
    Returns:
        int: Confidence score (0-100)
    """
    score = 0
    max_score = 100
    
    # Basic required fields (50%)
    if "app_id" in pricing_data and "app_name" in pricing_data:
        score += 10
    
    if "price_model_type" in pricing_data and pricing_data["price_model_type"]:
        score += 15
    
    if "has_free_tier" in pricing_data and "has_free_trial" in pricing_data:
        score += 10
    
    if "currency" in pricing_data:
        score += 15
    
    # Detailed pricing information (50%)
    if "pricing_tiers" in pricing_data and pricing_data["pricing_tiers"]:
        # More points for more tiers and price details
        tiers = pricing_data["pricing_tiers"]
        tier_score = min(len(tiers) * 5, 20)  # Up to 20 points for tiers
        
        # Check for price details in tiers
        price_details = 0
        for tier in tiers:
            if "monthly_price" in tier or "annual_price" in tier:
                price_details += 1
        
        # Normalize price details score
        if tiers:
            price_details_score = (price_details / len(tiers)) * 10
            tier_score += price_details_score
        
        score += tier_score
    
    if "usage_based_pricing" in pricing_data and pricing_data["usage_based_pricing"]:
        score += 15
    
    if "pricing_notes" in pricing_data and len(pricing_data.get("pricing_notes", "")) > 20:
        score += 5
    
    # Repair or extraction issues reduce confidence
    if pricing_data.get("json_repaired", False) or pricing_data.get("extraction_error", False):
        score = max(score - 20, 0)  # Reduce score by 20 points but not below 0
    
    # Return final score, capped at 100
    return min(score, max_score)

def run_bing_search(args):
    """Run pricing search using Azure Bing Search Grounding."""
    print(f"Starting Azure Bing Search Grounding for {args.app_name}...")
    
    # Run Bing search
    result = bing_grounding_search.main(args.app_name, args.app_slug)
    
    # Display the results
    if "error" in result:
        print(f"Search failed: {result['error']}")
    else:
        print("\nBing Search Results:")
        for i, text in enumerate(result["response_text"]):
            print(f"{text}\n")
        
        print("Citations:")
        for i, citation in enumerate(result["citations"]):
            print(f"{i+1}. [{citation['title']}]({citation['url']})")
        
        # Show extracted content from Jina.ai
        if "extracted_content" in result and result["extracted_content"]:
            print("\nExtracted Content and Pricing Analysis:")
            for i, content_item in enumerate(result["extracted_content"]):
                url = content_item.get("url", "Unknown URL")
                is_accessible = content_item.get("is_accessible", False)
                content = content_item.get("content")
                error = content_item.get("error")
                
                print(f"\n{i+1}. URL: {url}")
                print(f"   Accessible: {is_accessible}")
                
                if error:
                    print(f"   Error: {error}")
                
                if content:
                    # Print a preview of the content (first 200 chars)
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"   Content Preview: {content_preview}")
                    print(f"   Content Length: {len(content)} characters")
                
                # Display structured pricing data if available
                if "pricing_data" in content_item and content_item.get("pricing_analyzed", False):
                    pricing_data = content_item["pricing_data"]
                    print("\n   STRUCTURED PRICING DATA:")
                    
                    # Display pricing model types
                    if "price_model_type" in pricing_data and pricing_data["price_model_type"]:
                        print(f"   Pricing Model: {', '.join(pricing_data['price_model_type'])}")
                    
                    # Display free tier/trial info
                    if "has_free_tier" in pricing_data:
                        print(f"   Free Tier Available: {pricing_data['has_free_tier']}")
                    if "has_free_trial" in pricing_data:
                        print(f"   Free Trial Available: {pricing_data['has_free_trial']}")
                        if pricing_data.get("free_trial_period_days"):
                            print(f"   Free Trial Duration: {pricing_data['free_trial_period_days']} days")
                    
                    # Display currency
                    if "currency" in pricing_data:
                        print(f"   Currency: {pricing_data['currency']}")
                    
                    # Display pricing tiers
                    if "pricing_tiers" in pricing_data and pricing_data["pricing_tiers"]:
                        print("\n   Pricing Tiers:")
                        for j, tier in enumerate(pricing_data["pricing_tiers"]):
                            print(f"     {j+1}. {tier.get('tier_name', 'Unnamed Tier')}")
                            if "monthly_price" in tier:
                                print(f"        Monthly: {pricing_data.get('currency', '$')}{tier['monthly_price']}")
                            if "annual_price" in tier:
                                print(f"        Annual: {pricing_data.get('currency', '$')}{tier['annual_price']}")
                            if "features" in tier and tier["features"]:
                                print(f"        Features: {len(tier['features'])} features")
                    
                    # Display usage-based pricing
                    if "usage_based_pricing" in pricing_data and pricing_data["usage_based_pricing"]:
                        print("\n   Usage-Based Pricing:")
                        for j, metric in enumerate(pricing_data["usage_based_pricing"]):
                            print(f"     {j+1}. {metric.get('metric_name', 'Unnamed Metric')}")
                            if "unit" in metric:
                                print(f"        Unit: {metric['unit']}")
                            if "base_price" in metric:
                                print(f"        Base Price: {pricing_data.get('currency', '$')}{metric['base_price']}")
                    
                    # Display additional notes
                    if "pricing_notes" in pricing_data and pricing_data["pricing_notes"]:
                        print(f"\n   Notes: {pricing_data['pricing_notes'][:200]}...")
        
        # Highlight JSON results directory
        if "results_directory" in result:
            print(f"\nDetailed JSON results saved to: {result['results_directory']}")
    
    # Save results to file if requested via --output parameter
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Complete results saved to file: {args.output}")
    
    print("Bing search completed")

def run_indexing():
    """Create vector indexes for pricing data (placeholder for future implementation)."""
    print("Indexing functionality not yet implemented")
    print("This would create vector indexes in MongoDB for similarity search")

def main():
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Run the appropriate command
    if args.command == "discover":
        run_discovery(args)
    elif args.command == "extract":
        run_extraction(args)
    elif args.command == "pipeline":
        run_full_pipeline(args)
    elif args.command == "enhanced-pipeline":
        run_enhanced_pipeline(args)
    elif args.command == "bing":
        run_bing_search(args)
    elif args.command == "process-mongodb":
        run_mongodb_processing(args)
    elif args.command == "index":
        run_indexing()
    else:
        print("No command specified. Use --help to see available commands.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
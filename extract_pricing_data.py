import os
import json
import time
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from urllib.parse import quote_plus
import pricing_schema
from pricing_models import PricingData, PriceModelType
from typing import List, Optional
import asyncio
import logging
import uuid
import config
from pymongo import MongoClient, ReturnDocument
import openai
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = config.mongodb_uri
MONGODB_DB_NAME = config.mongodb_db_name

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = config.openai_endpoint
AZURE_OPENAI_API_KEY = config.openai_api_key
AZURE_OPENAI_API_VERSION = config.openai_api_version
AZURE_DEPLOYMENT = config.openai_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get MongoDB connection string from config
def get_mongodb_connection_string():
    # Get connection string from config
    connection_string = config.mongodb_uri
    
    # If not found, use default local connection
    if not connection_string:
        connection_string = "mongodb://localhost:27017/"
        logger.warning("MongoDB connection string not found, using default local connection")
    
    return connection_string

# Connect to MongoDB
def connect_to_mongodb():
    connection_string = get_mongodb_connection_string()
    client = MongoClient(connection_string)
    return client['apicus']

# Repair JSON if needed
def repair_json(json_str):
    """
    Attempt to repair malformed JSON by fixing common issues.
    
    Args:
        json_str (str): JSON string to repair
        
    Returns:
        Tuple[str, bool]: (Repaired JSON string, whether repair was needed)
    """
    repair_needed = False
    
    # Replace any triple quotes with single quotes
    if "'''" in json_str:
        json_str = json_str.replace("'''", "'")
        repair_needed = True
    
    # Handle None vs null - Python uses None, JSON uses null
    if " None," in json_str or ": None," in json_str or ": None}" in json_str:
        json_str = json_str.replace(" None,", " null,")
        json_str = json_str.replace(": None,", ": null,")
        json_str = json_str.replace(": None}", ": null}")
        repair_needed = True
    
    # Fix trailing commas in arrays and objects
    if re.search(r",\s*\]", json_str) or re.search(r",\s*\}", json_str):
        json_str = re.sub(r",(\s*\])", r"\1", json_str)
        json_str = re.sub(r",(\s*\})", r"\1", json_str)
        repair_needed = True
    
    # Fix missing commas in arrays
    if re.search(r"\[[^\[\]{}:,]*[^\[\]{}:,\s][^\[\]{}:,]*\s+[^\[\]{}:,]+", json_str):
        json_str = re.sub(r"(\[[^\[\]{}:,]*[^\[\]{}:,\s][^\[\]{}:,]*)\s+", r"\1, ", json_str)
        repair_needed = True
    
    # Fix unquoted property names
    if re.search(r"[\{\,]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", json_str):
        json_str = re.sub(r"([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_str)
        repair_needed = True
    
    # Fix boolean values
    if re.search(r":\s*True\b", json_str) or re.search(r":\s*False\b", json_str):
        json_str = re.sub(r":\s*True\b", r": true", json_str)
        json_str = re.sub(r":\s*False\b", r": false", json_str)
        repair_needed = True
    
    return json_str, repair_needed

def fetch_pricing_page_content(url):
    """
    Fetch the content of a pricing page using Jina.ai Reader.
    
    Args:
        url (str): URL of the pricing page
    
    Returns:
        tuple: (content, is_accessible) where content is the extracted text and is_accessible is a boolean
    """
    try:
        # Jina Reader API - this provides a clean, extracted version of the webpage
        reader_url = f"https://reader.jina.ai/api/extract?url={quote_plus(url)}"
        response = requests.get(reader_url)
        
        if response.status_code == 200:
            data = response.json()
            # Extract the main content text
            content = data.get("text", "")
            # Also get any tables that might contain pricing information
            tables = data.get("tables", [])
            table_text = "\n\n".join([f"Table {i+1}:\n" + table for i, table in enumerate(tables)])
            
            # Combine content and tables
            full_content = content
            if table_text:
                full_content += "\n\n" + table_text
            
            # Check if page was accessible with meaningful content
            is_accessible = len(full_content.strip()) > 100
                
            return full_content, is_accessible
        else:
            print(f"Error fetching content from {url}: HTTP {response.status_code}")
            return None, False
    except Exception as e:
        print(f"Error during content extraction for {url}: {e}")
        return None, False

def check_pricing_is_public(content):
    """
    Check if the pricing page contains actual public pricing information.
    
    Args:
        content (str): Text content from the pricing page
    
    Returns:
        bool: True if pricing appears to be public, False if it's likely behind a form or contact request
    """
    if not content:
        return False
    
    # Common phrases indicating pricing requires contacting sales
    contact_sales_indicators = [
        "contact sales",
        "contact us for pricing",
        "request a quote",
        "request quote",
        "get in touch",
        "talk to sales",
        "schedule a demo",
        "custom pricing",
        "custom quote",
        "speak with sales",
        "contact for enterprise",
        "tailored pricing",
        "pricing available upon request"
    ]
    
    # Check if the page content contains indicators that pricing is not public
    content_lower = content.lower()
    
    # Look for indicators suggesting pricing is not publicly available
    contact_sales_matches = [phrase for phrase in contact_sales_indicators if phrase in content_lower]
    
    # Look for price indicators (currency symbols or words like price/pricing)
    has_price_indicators = any(symbol in content for symbol in ['$', '€', '£', '¥']) or any(
        price_term in content_lower for price_term in ['per month', 'monthly', 'annually', '/mo', '/month', 'pricing tier'])
    
    # If we see both contact sales phrases and no price indicators, pricing is likely not public
    if contact_sales_matches and not has_price_indicators:
        return False
    
    # If we see price indicators, assume pricing is public
    if has_price_indicators:
        return True
    
    # If we can't determine, default to assuming it's not public
    return len(content.strip()) > 500  # If substantial content but no clear indicators, assume something is there

async def get_pricing_system_prompt():
    """Get the system prompt for pricing analysis with the schema from pricing_schema.py."""
    try:
        # Get schema for prompt from pricing_schema.py
        schema_json = pricing_schema.get_schema_for_prompt()
        
        # Build a system prompt with the schema
        system_prompt = """You are a pricing data extraction specialist. Your task is to analyze content from software product pricing pages and extract structured pricing information.

Extract the following details:
1. Pricing model types (subscription, usage-based, one-time, etc.)
2. Whether a free tier or free trial is available
3. Currency used
4. All pricing tiers with their names, monthly/annual prices, and features
5. Any usage-based pricing details
6. AI-specific pricing if applicable
7. Promotional offers
8. Additional fees

Format the information according to the exact schema provided below:

{}

If information is not available, omit those fields rather than guessing. Be precise with pricing values and clearly distinguish between monthly and annual pricing.

Do not include placeholder or example data - only extract what is explicitly stated in the content. Maintain the exact field names from the schema.""".format(schema_json)
        
        return system_prompt
    except Exception as e:
        logger.error(f"Error generating system prompt: {e}")
        # Return a basic fallback prompt
        return "Extract structured pricing information from this webpage content. Return results in JSON format."

async def fetch_pricing_page_content(url, headers=None, timeout=60):
    """Fetch content from a pricing page URL."""
    if not headers:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    logger.warning(f"URL {url} returned status code {response.status}")
                    return None
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type and 'application/json' not in content_type:
                    logger.warning(f"URL {url} has content type {content_type}, not HTML or JSON")
                    return None
                
                content = await response.text()
                return content
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

async def is_pricing_public(api_client, url_content, url):
    """Check if the pricing information is publicly available."""
    # If the content is empty or too short, consider pricing not public
    if not url_content or len(url_content) < 100:
        return False
    
    try:
        # Use OpenAI to determine if the page has public pricing
        response = await api_client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": "You are a pricing data analyzer. Your task is to determine if this webpage contains specific pricing information."},
                {"role": "user", "content": f"Does this webpage contain specific pricing information (prices, tiers, plans)? Please just answer YES or NO.\n\nURL: {url}\n\nContent: {url_content[:5000]}"}
            ],
            temperature=0,
            max_tokens=10
        )
        
        # Get the model's answer
        answer = response.choices[0].message.content.strip().upper()
        
        # Check if the answer contains YES
        return "YES" in answer
    except Exception as e:
        logger.error(f"Error checking if pricing is public for {url}: {e}")
        # Default to False on error
        return False

async def extract_pricing_data(api_client, app_id, app_name, pricing_content, product_type, prompt_version=2):
    logger.info(f"Analyzing pricing content for app_id: {app_id}, product_type: {product_type}")
    # Default max retries with exponential backoff
    max_retries = 3
    retry_count = 0
    backoff_factor = 2  # seconds

    while retry_count < max_retries:
        try:
            # Use Azure's OpenAI model to analyze the pricing content with structured outputs
            response = await api_client.chat.completions.create(
                model=config.openai_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": await get_pricing_system_prompt()},
                    {"role": "user", "content": pricing_content}
                ],
                temperature=0.2,
                max_tokens=4000,
                seed=1000,
                data_format="structured_outputs",
                schema=PricingData.schema(),
            )
            
            # Extract the structured pricing data from the model's response
            # This will be a PricingData object that has already been validated
            pricing_data = response.choices[0].message.data_format_output
            
            # Convert the Pydantic model to a dictionary for further processing
            pricing_dict = pricing_data.model_dump()
            
            # Ensure required fields are populated
            pricing_dict["app_id"] = app_id
            pricing_dict["app_name"] = app_name
            pricing_dict["analysis_date"] = datetime.utcnow().isoformat()
            pricing_dict["product_type"] = product_type
            
            # Convert Pydantic Enum values to strings for MongoDB storage
            if "price_model_type" in pricing_dict:
                pricing_dict["price_model_type"] = [
                    model_type.value if isinstance(model_type, PriceModelType) else model_type
                    for model_type in pricing_dict["price_model_type"]
                ]
            
            logger.info(f"Successfully extracted structured pricing data for app_id: {app_id}")
            return pricing_dict
            
        except Exception as e:
            logger.error(f"Error with structured output extraction: {e}")
            logger.info("Falling back to regular JSON extraction...")
            
            try:
                # Fallback to regular JSON extraction
                response = await api_client.chat.completions.create(
                    model=config.openai_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": await get_pricing_system_prompt()},
                        {"role": "user", "content": pricing_content}
                    ],
                    temperature=0.2,
                    max_tokens=4000,
                    seed=1000
                )
                
                # Extract the pricing data from the model's response
                content = response.choices[0].message.content
                
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    pattern = r"```json\s*([\s\S]*?)\s*```"
                    match = re.search(pattern, content)
                    if match:
                        content = match.group(1)
                elif "```" in content:
                    pattern = r"```\s*([\s\S]*?)\s*```"
                    match = re.search(pattern, content)
                    if match:
                        content = match.group(1)
                
                # Try to parse JSON, repair if needed
                try:
                    pricing_dict = json.loads(content.strip())
                    json_repaired = False
                except json.JSONDecodeError:
                    # Try repairing the JSON
                    repaired_json, json_repaired = repair_json(content.strip())
                    try:
                        pricing_dict = json.loads(repaired_json)
                    except json.JSONDecodeError as e:
                        # If repair fails, try to extract any JSON object from the text
                        logger.error(f"JSON repair failed: {e}")
                        pattern = r"\{[\s\S]*?\}"
                        match = re.search(pattern, content)
                        if match:
                            extracted_json, json_repaired = repair_json(match.group(0))
                            pricing_dict = json.loads(extracted_json)
                        else:
                            raise
                
                # Record if JSON repair was needed
                pricing_dict["json_repaired"] = json_repaired
                
                # Ensure required fields are populated
                pricing_dict["app_id"] = app_id
                pricing_dict["app_name"] = app_name
                pricing_dict["analysis_date"] = datetime.utcnow().isoformat()
                pricing_dict["product_type"] = product_type
                
                # Normalize price_model_type
                if "price_model_type" in pricing_dict:
                    # Ensure price_model_type is always a list
                    if pricing_dict["price_model_type"] is None:
                        pricing_dict["price_model_type"] = ["custom"]
                    elif not isinstance(pricing_dict["price_model_type"], list):
                        # If it's a string, convert to a single-item list
                        if isinstance(pricing_dict["price_model_type"], str):
                            pricing_dict["price_model_type"] = [pricing_dict["price_model_type"]]
                        else:
                            # Try to convert other types to string and then to a list
                            try:
                                pricing_dict["price_model_type"] = [str(pricing_dict["price_model_type"])]
                            except:
                                pricing_dict["price_model_type"] = ["custom"]
                else:
                    pricing_dict["price_model_type"] = ["custom"]

                # Normalize model types in the list
                if "price_model_type" in pricing_dict:
                    valid_types = [e.value for e in PriceModelType]
                    
                    # Map old model types to new enum values
                    model_type_mapping = {
                        "free": PriceModelType.FREE_TIER.value,
                        "freemium": PriceModelType.HYBRID.value,
                        "tiered": PriceModelType.SUBSCRIPTION.value,
                        "per_user": PriceModelType.SUBSCRIPTION.value,
                        "flat_rate": PriceModelType.ONE_TIME.value,
                        "contact_sales": PriceModelType.QUOTE_BASED.value,
                        "open_source": PriceModelType.FREE_TIER.value,
                        "unknown": PriceModelType.CUSTOM.value,
                    }
                    
                    for i, model_type in enumerate(pricing_dict["price_model_type"]):
                        if model_type is None:
                            pricing_dict["price_model_type"][i] = PriceModelType.CUSTOM.value
                        elif isinstance(model_type, str):
                            # Convert to lowercase and sanitize
                            model_type = model_type.lower().replace(' ', '_')
                            # Check if it matches one of our enum values
                            if model_type in model_type_mapping:
                                model_type = model_type_mapping[model_type]
                            elif model_type not in valid_types:
                                model_type = PriceModelType.CUSTOM.value
                            pricing_dict["price_model_type"][i] = model_type
                
                # Ensure array fields are not null but empty lists
                array_fields = [
                    'pricing_tiers',
                    'usage_based_pricing',
                    'promotional_offers',
                    'additional_fees',
                    'all_pricing_urls'
                ]
                
                for field in array_fields:
                    if field not in pricing_dict or pricing_dict[field] is None:
                        pricing_dict[field] = []
                
                # Create fallback pricing data in case extraction fails for some fields
                if 'has_free_tier' not in pricing_dict or pricing_dict['has_free_tier'] is None:
                    pricing_dict['has_free_tier'] = False
                    
                # Ensure ai_specific_pricing is a dict if present
                if 'ai_specific_pricing' in pricing_dict and pricing_dict['ai_specific_pricing'] is None:
                    pricing_dict['ai_specific_pricing'] = {}
                
                logger.info(f"Successfully extracted pricing data for app_id: {app_id} using fallback method")
                return pricing_dict
                
            except Exception as fallback_error:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to extract pricing data after {max_retries} attempts for app_id: {app_id}. Error: {fallback_error}")
                    # Create fallback pricing data in case extraction fails
                    fallback_data = {
                        "app_id": app_id,
                        "app_name": app_name,
                        "analysis_date": datetime.utcnow().isoformat(),
                        "product_type": product_type,
                        "price_model_type": [PriceModelType.CUSTOM.value],
                        "has_free_tier": False,
                        "pricing_tiers": [],
                        "usage_based_pricing": [],
                        "promotional_offers": [],
                        "additional_fees": [],
                        "all_pricing_urls": [],
                        "extraction_failed": True,
                        "error_message": str(fallback_error)
                    }
                    return fallback_data
                else:
                    # Calculate exponential backoff time
                    sleep_time = backoff_factor ** retry_count
                    logger.warning(f"Retry {retry_count} after error: {fallback_error}. Waiting {sleep_time} seconds before retry.")
                    await asyncio.sleep(sleep_time)

def create_embeddings(text, app_name):
    """
    Create vector embeddings for the pricing information using Azure OpenAI embedding model.
    
    Args:
        text (str): Text to create embeddings for
        app_name (str): App name for logging
    
    Returns:
        list: Vector embedding array
    """
    AZURE_EMBEDDING_ENDPOINT = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{os.getenv('AZURE_TEXT_EMBEDDING_3_SMALL_DEPLOYMENT')}/embeddings?api-version=2023-05-15"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    
    # Prepare text for embedding
    if isinstance(text, dict):
        # Convert dict to string
        text = json.dumps(text)
    
    payload = {
        "input": text[:8000],  # Truncate to avoid token limits
        "dimensions": 1536
    }
    
    try:
        response = requests.post(AZURE_EMBEDDING_ENDPOINT, headers=headers, json=payload)
        
        if response.status_code == 200:
            embedding_data = response.json()
            return embedding_data["data"][0]["embedding"]
        else:
            print(f"Error creating embeddings for {app_name}: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error during embedding creation for {app_name}: {e}")
        return None

def process_pricing_discovery_results(limit=10):
    """
    Process discovered pricing URLs and extract structured pricing data.
    
    Args:
        limit (int): Maximum number of apps to process
    """
    db = connect_to_mongodb()
    
    # Get apps with discovered pricing URLs
    discovery_collection = db["apicus-apps-prices-discovery"]
    pricing_collection = db["apicus-apps-prices"]
    
    # Find records with pricing_url that haven't been processed yet
    query = {
        "$or": [
            {"primary_pricing_url": {"$ne": None}},
            {"pricing_urls": {"$ne": [], "$exists": True}}
        ],
        "processed": {"$ne": True}
    }
    
    projection = {
        "app_id": 1,
        "app_name": 1,
        "app_slug": 1,
        "pricing_urls": 1,
        "primary_pricing_url": 1
    }
    
    cursor = discovery_collection.find(query, projection).limit(limit)
    discovery_results = list(cursor)
    
    print(f"Found {len(discovery_results)} apps with pricing URLs to process")
    
    for result in discovery_results:
        app_id = result["app_id"]
        app_name = result["app_name"]
        app_slug = result.get("app_slug", "")
        
        # Get all pricing URLs, with primary URL first if available
        pricing_urls = []
        if "primary_pricing_url" in result and result["primary_pricing_url"]:
            pricing_urls.append(result["primary_pricing_url"])
        
        # Add other URLs that aren't already included
        if "pricing_urls" in result and result["pricing_urls"]:
            for url in result["pricing_urls"]:
                if url not in pricing_urls:
                    pricing_urls.append(url)
        
        if not pricing_urls:
            print(f"No valid pricing URLs found for {app_name}, skipping")
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"processed": True, "error": "No valid pricing URLs found"}}
            )
            continue
        
        print(f"Processing pricing for: {app_name} (found {len(pricing_urls)} URLs)")
        
        # Check if we already have pricing data for this app
        existing = pricing_collection.find_one({"app_id": app_id})
        if existing:
            print(f"Already have pricing data for {app_name}, skipping")
            # Mark as processed in discovery collection
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"processed": True}}
            )
            continue
        
        # Try each URL in order until we get valid content
        content = None
        is_page_accessible = False
        successful_url = None
        
        for pricing_url in pricing_urls:
            print(f"  Trying URL: {pricing_url}")
            content, is_page_accessible = fetch_pricing_page_content(pricing_url)
            
            if content and is_page_accessible:
                successful_url = pricing_url
                print(f"  Successfully retrieved content from: {pricing_url}")
                break
            else:
                print(f"  Failed to retrieve content from: {pricing_url}")
        
        if not content or not successful_url:
            print(f"Could not fetch content from any URL for {app_name}, skipping")
            # Create a minimal document to reflect that pricing page was not accessible
            minimal_doc = pricing_schema.create_empty_pricing_doc(app_id, app_name, app_slug, pricing_urls[0])
            minimal_doc["pricing_page_accessible"] = False
            minimal_doc["is_pricing_public"] = False
            minimal_doc["pricing_notes"] = f"Attempted {len(pricing_urls)} URLs but none were accessible during scraping."
            minimal_doc["all_pricing_urls"] = pricing_urls
            
            try:
                pricing_collection.insert_one(minimal_doc)
                print(f"Stored minimal pricing data for {app_name} (pages not accessible)")
            except Exception as e:
                print(f"Error storing minimal pricing data for {app_name}: {e}")
            
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"error": "Failed to fetch content from any URL", "processed": True}}
            )
            continue
        
        # Check if pricing is public
        is_pricing_public = check_pricing_is_public(content)
        
        # Analyze content with OpenAI using structured outputs
        pricing_data = analyze_pricing_with_openai(content, app_name, app_slug, is_pricing_public, is_page_accessible)
        
        if not pricing_data:
            print(f"Could not extract pricing data for {app_name}, creating minimal document")
            # Create a minimal document for apps where extraction failed
            minimal_doc = pricing_schema.create_empty_pricing_doc(app_id, app_name, app_slug, successful_url)
            minimal_doc["pricing_page_accessible"] = is_page_accessible
            minimal_doc["is_pricing_public"] = is_pricing_public
            minimal_doc["pricing_notes"] = "Extraction failed, but pricing page was accessible."
            minimal_doc["raw_pricing_text"] = content
            minimal_doc["all_pricing_urls"] = pricing_urls
            
            try:
                pricing_collection.insert_one(minimal_doc)
                print(f"Stored minimal pricing data for {app_name} (extraction failed)")
            except Exception as e:
                print(f"Error storing minimal pricing data for {app_name}: {e}")
            
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"error": "Failed to extract pricing data", "processed": True}}
            )
            continue
        
        # Create embeddings for search
        rich_text = f"{app_name} {app_slug} {pricing_data.get('price_model_type', [])} {pricing_data.get('currency', '')} "
        if "pricing_tiers" in pricing_data and pricing_data["pricing_tiers"]:
            tier_names = [tier.get('tier_name', '') for tier in pricing_data.get('pricing_tiers', [])]
            rich_text += f"tiers: {', '.join(tier_names)} "
        
        if "pricing_notes" in pricing_data and pricing_data["pricing_notes"]:
            rich_text += pricing_data["pricing_notes"]
            
        embedding_vector = create_embeddings(rich_text, app_name)
        
        # Prepare document for MongoDB
        pricing_doc = pricing_schema.create_empty_pricing_doc(app_id, app_name, app_slug, successful_url)
        
        # Add all pricing URLs we tried
        pricing_doc["all_pricing_urls"] = pricing_urls
        
        # Update with extracted data
        pricing_doc.update(pricing_data)
        
        # Add embeddings if available
        if embedding_vector:
            pricing_doc["embedding_vector"] = embedding_vector
            
        # Add raw content for reference
        pricing_doc["raw_pricing_text"] = content
        
        # Store in MongoDB
        try:
            pricing_collection.insert_one(pricing_doc)
            print(f"Stored pricing data for {app_name}")
            
            # Mark as processed in discovery collection
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"processed": True, "success": True}}
            )
        except Exception as e:
            print(f"Error storing pricing data for {app_name}: {e}")
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": {"error": str(e), "processed": True}}
            )
        
        # Respect rate limits
        time.sleep(1)

if __name__ == "__main__":
    # Process a limited number of apps for testing
    process_pricing_discovery_results(limit=5) 
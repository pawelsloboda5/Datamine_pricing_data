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

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")  # Updated to use latest API version
# Changed from o3 model to o4 model for better function calling support
AZURE_DEPLOYMENT = os.getenv("AZURE_o4_DEPLOYMENT")  # Using o4 model instead of o3 for better function calling

def connect_to_mongodb():
    """Connect to MongoDB and return database client."""
    try:
        client = pymongo.MongoClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        print(f"Connected to MongoDB: {MONGODB_DB_NAME}")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

# Add the improved repair_json function
def repair_json(json_str):
    """
    Attempt to repair malformed JSON strings.
    
    Args:
        json_str (str): Potentially malformed JSON string
    
    Returns:
        str: Repaired JSON string or None if repair failed
    """
    try:
        # Check if it's already valid
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass
    
    # Remove any text before the first { and after the last }
    first_brace = json_str.find('{')
    last_brace = json_str.rfind('}')
    if first_brace != -1 and last_brace != -1:
        json_str = json_str[first_brace:last_brace+1]
    
    # More robust repair for JSON issues
    try:
        # First, check for common field errors that might be causing problems
        if '"all_pricing_urls": [' in json_str and '"],' not in json_str and '"]' not in json_str:
            # Fix the common case where all_pricing_urls array is not terminated
            pattern = r'"all_pricing_urls": \[(.*?)(?=,\s*"|\s*\})'
            match = re.search(pattern, json_str, re.DOTALL)
            if match:
                url_content = match.group(1).strip()
                # If the array content doesn't end with a quote and bracket, add them
                if not url_content.endswith(']'):
                    if not url_content.endswith('"'):
                        # If the last item doesn't end with a quote, add it
                        if url_content and not url_content.endswith('"'):
                            url_content += '"'
                    # Close the array
                    fixed_content = f'"all_pricing_urls": [{url_content}]'
                    json_str = json_str.replace(match.group(0), fixed_content)

        # Simple repair for unclosed quotes
        lines = json_str.split('\n')
        repaired_lines = []
        
        in_string = False  # Track if we're inside a string across lines
        for i, line in enumerate(lines):
            # Check for quote pairs in this line
            for char_index, char in enumerate(line):
                if char == '"' and (char_index == 0 or line[char_index-1] != '\\'):
                    in_string = not in_string
            
            # If we're still in a string at the end of the line, close it
            if in_string:
                line = line + '"'
                in_string = False
            
            # Check for lines that should end with comma but don't
            if i < len(lines) - 1:  # Not the last line
                line_stripped = line.strip()
                if (line_stripped.endswith('"') or 
                    line_stripped.endswith('}') or 
                    line_stripped.endswith(']')) and not line_stripped.endswith(','):
                    # Check if next line starts an object/array/key
                    next_line = lines[i+1].strip()
                    if next_line.startswith('"') or next_line.startswith('{') or next_line.startswith('['):
                        line = line.rstrip() + ','
                        
            repaired_lines.append(line)
        
        # Try to parse the repaired JSON
        repaired_json = '\n'.join(repaired_lines)
        
        # Additional cleanup for common JSON issues
        # 1. Fix arrays with trailing commas
        repaired_json = re.sub(r',\s*]', ']', repaired_json)
        # 2. Fix objects with trailing commas
        repaired_json = re.sub(r',\s*}', '}', repaired_json)
        
        try:
            json.loads(repaired_json)
            print("Successfully repaired JSON")
            return repaired_json
        except json.JSONDecodeError as e:
            print(f"First repair attempt failed: {e}")
            
            # If our first attempt failed, try a more targeted approach
            # Attempt to create a valid minimal object with the required fields
            try:
                # Extract the fields we can clearly identify
                app_id_match = re.search(r'"app_id":\s*"([^"]+)"', json_str)
                app_name_match = re.search(r'"app_name":\s*"([^"]+)"', json_str)
                app_slug_match = re.search(r'"app_slug":\s*"([^"]+)"', json_str)
                pricing_url_match = re.search(r'"pricing_url":\s*"([^"]+)"', json_str)
                
                minimal_json = {
                    "app_id": app_id_match.group(1) if app_id_match else "unknown",
                    "app_name": app_name_match.group(1) if app_name_match else "Unknown App",
                    "app_slug": app_slug_match.group(1) if app_slug_match else "unknown-app",
                    "pricing_url": pricing_url_match.group(1) if pricing_url_match else "",
                    "all_pricing_urls": [],
                    "price_model_type": ["unknown"],
                    "has_free_tier": False,
                    "has_free_trial": False,
                    "currency": "USD",
                    "is_pricing_public": True,
                    "pricing_page_accessible": True,
                    "pricing_notes": "JSON repair recovered only partial data due to malformed response.",
                    "extraction_timestamp": datetime.now().isoformat(),
                    "repair_attempted": True
                }
                
                print("Created minimal valid JSON object with extracted fields")
                return json.dumps(minimal_json)
            except Exception as e2:
                print(f"Minimal JSON creation failed: {e2}")
                return None
    except Exception as e:
        print(f"Error during JSON repair: {e}")
        return None

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

def analyze_pricing_with_openai(content, app_name, app_slug, is_pricing_public, is_page_accessible):
    """
    Use Azure OpenAI to analyze pricing page content and extract structured data
    using Pydantic models for structured outputs.
    
    Args:
        content (str): Text content from the pricing page
        app_name (str): Name of the app
        app_slug (str): URL-friendly version of app name
        is_pricing_public (bool): Whether the pricing appears to be publicly available
        is_page_accessible (bool): Whether the pricing page was accessible
    
    Returns:
        dict: Structured pricing data based on our schema
    """
    if not content or len(content.strip()) < 50:
        print(f"Insufficient content for {app_name} to analyze")
        return None
    
    # Initialize Azure OpenAI client
    from openai import AzureOpenAI
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    
    # Build the system prompt
    system_prompt = f"""You are an expert at analyzing pricing pages for software products and extracting structured pricing information.
Your task is to extract pricing details for '{app_name}' from the provided content.

Extract as much information as you can about:
1. Pricing tiers (e.g., Basic, Pro, Enterprise)
2. Monthly and annual pricing
3. Free trials or free tiers
4. Usage-based pricing (especially for API or AI services)
5. Features included in each tier
6. Any special promotional offers
7. Any limitations on usage (users, storage, etc.)

Important notes:
- If specific information is not available, omit those fields rather than guessing
- For price_model_type, identify all applicable pricing models
- If limits appear to be unlimited, use the string 'unlimited' rather than a number
- Include any pricing_notes that might help explain unusual or complex pricing structures
- Set is_pricing_public to {str(is_pricing_public).lower()} based on our analysis
- Set pricing_page_accessible to {str(is_page_accessible).lower()} based on our analysis
"""

    # Build user prompt - limit content length to avoid token limits
    max_content_length = 32000  # Limiting to ~32k chars
    if len(content) > max_content_length:
        truncated_content = content[:max_content_length] + "...[content truncated due to length]"
        user_prompt = f"Extract pricing information for {app_name} from this content:\n\n{truncated_content}"
    else:
        user_prompt = f"Extract pricing information for {app_name} from this content:\n\n{content}"

    # Multiple retries with backoff
    max_retries = 3
    backoff_time = 2  # seconds
    
    for retry in range(max_retries):
        try:
            print(f"Analyzing pricing content for {app_name} using {AZURE_DEPLOYMENT} model (attempt {retry+1}/{max_retries})...")
            
            # Use parse method with Pydantic model for structured output
            completion = client.beta.chat.completions.parse(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PricingData,
                temperature=0.1,
                seed=42  # For consistent results
            )
            
            # Extract the structured data
            pricing_data = completion.choices[0].message.parsed
            
            # Convert to dictionary for backward compatibility
            pricing_dict = pricing_data.model_dump()
            
            # Ensure required fields are present
            if app_slug and not pricing_dict.get("app_slug"):
                pricing_dict["app_slug"] = app_slug
                
            if not pricing_dict.get("app_id"):
                pricing_dict["app_id"] = app_slug.lower() if app_slug else app_name.lower().replace(" ", "_")
                
            # Add timestamp for when this extraction was performed
            pricing_dict["extraction_timestamp"] = datetime.now().isoformat()
            
            # Convert PriceModelType enum values to strings
            if "price_model_type" in pricing_dict:
                pricing_dict["price_model_type"] = [str(model_type) for model_type in pricing_dict["price_model_type"]]
            
            return pricing_dict
            
        except Exception as e:
            print(f"Error calling Azure OpenAI structured outputs API for {app_name}: {e}")
            
            # Log the error for debugging
            error_log_file = f"error_logs_{app_name.lower().replace(' ', '_')}.txt"
            with open(error_log_file, "w", encoding="utf-8") as f:
                f.write(f"Structured Output Error: {e}\n\n")
            
            # Only retry if we haven't reached max retries
            if retry < max_retries - 1:
                wait_time = backoff_time * (2 ** retry)
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
                # Simplify the prompt for the next retry
                if retry == max_retries - 2:  # Last retry
                    print("Simplifying prompt for final retry...")
                    system_prompt = f"""Extract basic pricing information for '{app_name}'. 
                    Focus on core pricing details: pricing tiers, prices, and free tier availability."""
                    
                    # Reduce content length further
                    max_content_length = max_content_length // 2
                    user_prompt = f"Extract pricing information for {app_name}:\n\n{content[:max_content_length]}"
            else:
                # If we've exhausted all retries, try the legacy approach
                print(f"Structured output failed after {max_retries} attempts, falling back to legacy approach")
                return create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible)
    
    # If we've reached here, all retries failed
    return create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible)

def create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible):
    """
    Create a minimal pricing data document when extraction fails.
    
    Args:
        app_name (str): Name of the app
        app_slug (str): URL-friendly version of app name
        is_pricing_public (bool): Whether the pricing appears to be publicly available
        is_page_accessible (bool): Whether the pricing page was accessible
        
    Returns:
        dict: Minimal pricing data
    """
    return {
        "app_id": app_slug.lower() if app_slug else app_name.lower().replace(" ", "_"),
        "app_name": app_name,
        "app_slug": app_slug,
        "price_model_type": ["unknown"],
        "has_free_tier": False,
        "has_free_trial": False,
        "currency": "USD",
        "is_pricing_public": is_pricing_public,
        "pricing_page_accessible": is_page_accessible,
        "pricing_notes": "Failed to extract structured pricing information after multiple attempts.",
        "extraction_error": True,
        "extraction_timestamp": datetime.now().isoformat()
    }

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
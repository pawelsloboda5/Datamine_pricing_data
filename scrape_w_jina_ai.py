import os
import requests
import json
import time
from datetime import datetime
from urllib.parse import quote, urlparse, quote_plus
from dotenv import load_dotenv
import pymongo
from pymongo.errors import DuplicateKeyError

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME")

# Jina.ai configuration
JINA_SEARCH_ENDPOINT = "https://reader.jina.ai/api/extract"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_4O_MINI_DEPLOYMENT = os.getenv("Azure_4o_MINI_DEPLOYMENT", "gpt-4o-mini-apicus")

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

def fetch_content_with_jina(url):
    """
    Fetch content from a URL using Jina.ai Reader API.
    
    Args:
        url (str): URL to extract content from
    
    Returns:
        tuple: (content, is_accessible) where content is the extracted text and is_accessible is a boolean
    """
    try:
        # Jina Reader API - provides a clean, extracted version of the webpage
        reader_url = f"{JINA_SEARCH_ENDPOINT}?url={quote_plus(url)}"
        headers = {'Accept': 'application/json'}
        
        print(f"Fetching content using Jina.ai Reader API: {url}")
        response = requests.get(reader_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # For debugging, print the response structure
                if "data" in data:
                    # Extract the main content text from the nested structure
                    content = data["data"].get("content", "")
                    title = data["data"].get("title", "")
                    
                    # If we got title but no content, create minimal content
                    if title and not content:
                        content = f"# {title}\n\n"
                else:
                    # Extract the main content text directly
                    content = data.get("text", "")
                    
                    # Also get any tables that might contain pricing information
                    tables = data.get("tables", [])
                    table_text = "\n\n".join([f"Table {i+1}:\n" + table for i, table in enumerate(tables)])
                    
                    # Combine content and tables
                    if content and table_text:
                        content += "\n\n" + table_text
                
                # Check if page was accessible with meaningful content
                is_accessible = len(content.strip()) > 100
                
                if is_accessible:
                    print(f"Successfully extracted {len(content)} characters from {url}")
                    return content, is_accessible
                else:
                    print(f"Insufficient content extracted from {url}")
                    return None, False
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response from Jina.ai for {url}: {e}")
                return None, False
        else:
            print(f"Error fetching content from {url}: HTTP {response.status_code}")
            return None, False
    except Exception as e:
        print(f"Error during content extraction for {url}: {e}")
        return None, False

def search_pricing_page(app_name, app_slug=None):
    """
    This function previously used Jina.ai for search, but now acts as a placeholder.
    For actual search functionality, use the Bing Search implementation from bing_grounding_search.py.
    
    Args:
        app_name (str): Name of the app
        app_slug (str, optional): URL-friendly version of app name
    
    Returns:
        dict: Search results with pricing page information
    """
    # Construct likely pricing URLs based on app information
    pricing_urls = []
    
    if app_slug:
        # If we have a slug, create some common pricing URL patterns
        base_url = f"https://{app_slug}.com"
        pricing_urls = [
            f"{base_url}/pricing",
            f"{base_url}/plans",
            f"{base_url}/pricing-plans",
            f"{base_url}/subscription",
            f"{base_url}/products/pricing"
        ]
    
    return {
        "app_name": app_name,
        "app_slug": app_slug,
        "search_term": f"{app_name} pricing",
        "search_results": [],  # No search results - we're not using Jina.ai for search anymore
        "constructed_urls": pricing_urls,  # Provide constructed URLs as an alternative
        "result_count": 0,
        "timestamp": datetime.now().isoformat()
    }

def analyze_search_results_with_gpt4o(search_results, app_name, app_slug):
    """
    Use Azure GPT-4o mini to analyze search results and identify the top 3 most likely pricing pages.
    
    Args:
        search_results (dict): Search results from Jina.ai
        app_name (str): Name of the app
        app_slug (str): URL-friendly version of app name
    
    Returns:
        list: List of dictionaries containing the top 3 URLs and confidence scores
    """
    if "error" in search_results or not search_results.get("search_results"):
        print(f"No valid search results for {app_name} to analyze with GPT-4o")
        return []
    
    results = search_results["search_results"]
    
    if not results:
        print(f"Empty search results list for {app_name}")
        return []
    
    # Prepare the prompt for GPT-4o mini
    prompt = f"""Analyze these search results for "{app_name}" pricing pages and identify the top 3 most likely official pricing pages.

App Name: {app_name}
App Slug: {app_slug or 'Unknown'}

Search Results:
"""
    
    # Add search results to the prompt with more detailed context
    for i, result in enumerate(results[:10]):  # Limit to first 10 results
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        snippet = result.get("snippet", "No snippet")[:200]  # Limit snippet length
        prompt += f"\n{i+1}. Title: {title}\n   URL: {url}\n   Snippet: {snippet}\n"
    
    prompt += """
Return the top 3 URLs that are most likely to be official pricing pages for this app. For each URL, include a confidence score (0-100%) and a brief explanation.

When evaluating the URLs, consider these points:
1. Official pricing pages often contain keywords like "pricing", "plans", "subscriptions", etc.
2. The URL should be from the official domain of the app when possible
3. Look for pages that list different pricing tiers, features, or subscription options
4. Pages with pricing tables or comparison charts are highly relevant
5. Blog posts or review sites discussing pricing are less relevant than official pages

Your response should be in this JSON format:
[
  {
    "url": "URL1",
    "confidence": 95,
    "explanation": "This is the official pricing page because..."
  },
  {
    "url": "URL2",
    "confidence": 80,
    "explanation": "This appears to be..."
  },
  {
    "url": "URL3",
    "confidence": 65,
    "explanation": "This might be..."
  }
]

Only return valid JSON, no additional text.
"""
    
    # Call Azure OpenAI API
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert at analyzing search results to find official pricing pages for software applications. Your task is to identify the most likely URLs for pricing pages and return them in JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800,
        "response_format": {"type": "json_object"}
    }
    
    api_url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_4O_MINI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    
    retry_count = 0
    max_retries = 2
    backoff_time = 2  # seconds
    
    while retry_count <= max_retries:
        try:
            print(f"Calling Azure OpenAI API for {app_name} (attempt {retry_count + 1})")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse the JSON response
                try:
                    content_dict = json.loads(content)
                    
                    # Check if we got the expected array format
                    if isinstance(content_dict, dict) and "array" in content_dict:
                        pricing_pages = content_dict["array"]
                    elif isinstance(content_dict, list):
                        pricing_pages = content_dict
                    else:
                        # Try to find any array in the response
                        for key, value in content_dict.items():
                            if isinstance(value, list) and len(value) > 0:
                                pricing_pages = value
                                break
                        else:
                            print(f"Unexpected response format for {app_name}: {content[:100]}...")
                            pricing_pages = []
                    
                    # Validate each entry has the required fields
                    validated_pages = []
                    for page in pricing_pages:
                        if isinstance(page, dict) and "url" in page:
                            if "confidence" not in page:
                                page["confidence"] = 50  # Default confidence
                            if "explanation" not in page:
                                page["explanation"] = "No explanation provided"
                            validated_pages.append(page)
                    
                    return validated_pages
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing GPT-4o response for {app_name}: {e}")
                    print(f"Response: {content[:200]}...")
                    # Try to extract URLs directly from content as a last resort
                    if retry_count == max_retries:
                        return extract_urls_from_text(content, app_name)
            elif response.status_code == 429:
                # Rate limit hit, wait longer before retry
                retry_count += 1
                wait_time = backoff_time * (2 ** retry_count)
                print(f"Rate limit hit for {app_name}, waiting {wait_time}s before retry")
                time.sleep(wait_time)
                continue
            else:
                print(f"Azure OpenAI API error for {app_name}: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
            # If we got here, either we got a non-429 error or already processed results
            break
                
        except requests.exceptions.RequestException as e:
            print(f"Network error calling Azure OpenAI API for {app_name}: {e}")
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = backoff_time * (2 ** retry_count)
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                break
        except Exception as e:
            print(f"Unexpected error analyzing search results for {app_name}: {e}")
            break
    
    # If we've exhausted retries or had another error
    return []

def extract_urls_from_text(text, app_name):
    """
    Last resort function to extract URLs from text when JSON parsing fails.
    
    Args:
        text (str): The text containing potential URLs
        app_name (str): Name of the app for logging
    
    Returns:
        list: List of dictionaries with URLs
    """
    import re
    
    print(f"Attempting to extract URLs directly from text for {app_name}")
    
    # Simple regex to find URLs
    url_pattern = r'https?://[^\s,"\'\)\}]+'
    urls = re.findall(url_pattern, text)
    
    result = []
    for i, url in enumerate(urls[:3]):  # Take top 3
        result.append({
            "url": url,
            "confidence": 50 - (i * 10),  # Decreasing confidence
            "explanation": "URL extracted from text response"
        })
    
    return result

def normalize_url(url):
    """
    Normalize and validate a URL to ensure it's properly formatted.
    
    Args:
        url (str): The URL to normalize
    
    Returns:
        str: The normalized URL or None if invalid
    """
    if not url:
        return None
        
    # Remove whitespace and quotes
    url = url.strip().strip('"\'')
    
    # Ensure URL has a scheme
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Parse and normalize components
        parsed = urlparse(url)
        normalized = parsed.geturl()
        
        # Simple validation - must have domain with at least one dot
        if not parsed.netloc or '.' not in parsed.netloc:
            return None
            
        return normalized
    except Exception:
        return None

def extract_pricing_url(search_results):
    """
    Extract the top 3 most likely pricing page URLs from search results using GPT-4o mini.
    
    Args:
        search_results (dict): Search results from Jina.ai
    
    Returns:
        list: Top 3 most likely pricing page URLs, or empty list if none found
    """
    if "error" in search_results or not search_results.get("search_results"):
        print(f"No valid search results found: {'error' in search_results and search_results['error'] or 'Empty results'}")
        return []
    
    app_name = search_results["app_name"]
    app_slug = ""  # Extract from search term if available
    
    search_term = search_results.get("search_term", "")
    if search_term and " " in search_term:
        # If search term is like "App Name app-slug.com", extract app-slug
        parts = search_term.split()
        if len(parts) > 1 and ".com" in parts[-1]:
            app_slug = parts[-1].replace(".com", "")
    
    # Get analysis from GPT-4o mini
    pricing_pages = analyze_search_results_with_gpt4o(search_results, app_name, app_slug)
    
    if not pricing_pages:
        print(f"GPT-4o analysis failed for {app_name}, falling back to rule-based extraction")
        return fallback_extract_pricing_url(search_results)
    
    # Extract URLs and confidence scores for logging
    result_urls = []
    for page in pricing_pages:
        if isinstance(page, dict) and "url" in page and "confidence" in page:
            url = page.get("url")
            # Normalize and validate the URL
            normalized_url = normalize_url(url)
            
            if normalized_url:  # Only add valid URLs
                result_urls.append({
                    "url": normalized_url,
                    "confidence": page.get("confidence", 0),
                    "explanation": page.get("explanation", "No explanation provided")
                })
    
    # Sort by confidence score (highest first) if not already sorted
    result_urls.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Extract just the URLs for return value
    urls = [item["url"] for item in result_urls[:3]]
    
    # Log the results
    print(f"GPT-4o found {len(urls)} potential pricing pages for {app_name}")
    for i, page in enumerate(result_urls[:3]):
        print(f"  {i+1}. {page['url']} (Confidence: {page['confidence']}%)")
        print(f"     Reason: {page['explanation'][:100]}...")
    
    return urls  # Return top URLs (up to 3)

def fallback_extract_pricing_url(search_results):
    """
    Fallback method to extract pricing URL using rule-based approach.
    
    Args:
        search_results (dict): Search results from Jina.ai
    
    Returns:
        list: List containing the most likely pricing page URL, or empty list if none found
    """
    if "error" in search_results or not search_results.get("search_results"):
        return []
    
    results = search_results["search_results"]
    app_name = search_results["app_name"].lower()
    
    # Store potential URLs with a score
    potential_urls = []
    
    # Common patterns for pricing pages
    pricing_indicators = [
        "/pricing", "/plans", "/subscription", "-pricing", 
        "-plans", "pricing.html", "plans.html", "price"
    ]
    
    for result in results:
        url = result.get("url", "")
        title = result.get("title", "").lower()
        
        score = 0
        
        # Check for pricing indicators in URL
        for indicator in pricing_indicators:
            if indicator in url.lower():
                score += 3
                break
        
        # Check for pricing terms in title
        if "pricing" in title:
            score += 3
        elif "plans" in title or "subscription" in title:
            score += 2
        
        # Check if domain seems related to app name
        if app_name in url.lower():
            score += 4
        elif app_name.replace(" ", "") in url.lower():
            score += 3
        
        # If the URL has a reasonable score, add it to potential URLs
        if score >= 3:
            potential_urls.append({"url": url, "score": score})
    
    # Sort by score, highest first
    potential_urls.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top URLs (or empty list if none found)
    return [item["url"] for item in potential_urls[:3]] if potential_urls else []

def get_apps_from_mongodb(db, limit=None):
    """
    Retrieve apps from MongoDB for processing.
    
    Args:
        db: MongoDB database connection
        limit (int, optional): Limit the number of apps to retrieve
    
    Returns:
        list: List of app documents
    """
    collection = db["apicus-processed-apps"]
    
    # Query to get apps that don't already have pricing data
    query = {
        # We could add filters here to prioritize specific apps
    }
    
    projection = {
        "app_id": 1,
        "name": 1,
        "slug": 1,
        "description": 1
    }
    
    cursor = collection.find(query, projection)
    
    if limit:
        cursor = cursor.limit(limit)
    
    return list(cursor)

def process_apps(limit=10):
    """
    Main function to process apps and find their pricing pages.
    
    Args:
        limit (int): Maximum number of apps to process
    """
    db = connect_to_mongodb()
    
    # Get apps from MongoDB
    apps = get_apps_from_mongodb(db, limit)
    print(f"Retrieved {len(apps)} apps for processing")
    
    # Create results collection if it doesn't exist
    if "apicus-apps-prices-discovery" not in db.list_collection_names():
        db.create_collection("apicus-apps-prices-discovery")
    
    discovery_collection = db["apicus-apps-prices-discovery"]
    
    # Process each app
    for app in apps:
        app_name = app["name"]
        app_slug = app.get("slug")
        app_id = app["app_id"]
        
        print(f"Processing: {app_name} ({app_slug})")
        
        # Check if we already processed this app
        existing = discovery_collection.find_one({"app_id": app_id})
        if existing:
            print(f"Already processed {app_name}, skipping")
            continue
        
        # Get potential pricing URLs
        search_results = search_pricing_page(app_name, app_slug)
        
        # For backward compatibility, we'll keep the extract_pricing_url function
        # but adapt our process to work with the new approach
        
        # Instead of using search-based extraction, use constructed URLs or fallback
        pricing_urls = search_results.get("constructed_urls", [])
        
        # If we don't have constructed URLs, use the fallback approach
        if not pricing_urls:
            print(f"No constructed URLs for {app_name}, using fallback approach")
            pricing_urls = fallback_extract_pricing_url(search_results)
        
        # Try to fetch content from each URL until we find one that works
        valid_pricing_urls = []
        for url in pricing_urls:
            content, is_accessible = fetch_content_with_jina(url)
            if content and is_accessible:
                valid_pricing_urls.append(url)
                print(f"Found valid pricing page at {url}")
                break
        
        # Store results
        result_doc = {
            "app_id": app_id,
            "app_name": app_name,
            "app_slug": app_slug,
            "search_results": search_results,
            "pricing_urls": pricing_urls,
            "valid_pricing_urls": valid_pricing_urls,
            "primary_pricing_url": valid_pricing_urls[0] if valid_pricing_urls else None,
            "processed_at": datetime.now()
        }
        
        try:
            discovery_collection.insert_one(result_doc)
            print(f"Stored discovery results for {app_name}")
        except DuplicateKeyError:
            print(f"Duplicate entry for {app_name}, updating instead")
            discovery_collection.update_one(
                {"app_id": app_id},
                {"$set": result_doc}
            )
        
        # Respect rate limits
        time.sleep(1)

if __name__ == "__main__":
    # Process a limited number of apps for testing
    process_apps(limit=10)

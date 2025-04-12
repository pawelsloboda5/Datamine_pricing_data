#!/usr/bin/env python3
"""
Azure Bing Search Grounding integration for the Apicus pricing discovery pipeline.
This module provides capabilities to perform up-to-date web searches using Azure AI Agent Service.
"""

import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient  
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import BingGroundingTool
import requests
from urllib.parse import quote_plus
import pricing_schema
from pricing_models import PricingData, PriceModelType
from typing import List, Dict, Optional
from extract_pricing_data import check_pricing_is_public
from scrape_w_jina_ai import fetch_content_with_jina

# Load environment variables
load_dotenv()

# Azure AI Agent Service configuration
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME", "apicus-bing-connection")
MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_o4_DEPLOYMENT", "apicus-gpt-4o")  
API_VERSION = "2024-12-01-preview"  # Required API version for Bing grounding

# Azure OpenAI configuration for pricing extraction
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")  # Updated to current API version
AZURE_DEPLOYMENT = os.getenv("AZURE_o4_DEPLOYMENT", "o4-apicus")  # Using o4 model that supports structured outputs

def initialize_project_client():
    """Initialize and return the Azure AI Project client."""
    try:
        # Create credential with explicit exclusions to avoid proxy settings
        credential = DefaultAzureCredential(
            exclude_shared_token_cache_credential=True,
            exclude_visual_studio_code_credential=True,
            exclude_powershell_credential=True,
            exclude_environment_credential=True
        )
        
        # Initialize the client with minimal credential options
        project_client = AIProjectClient.from_connection_string(
            credential=credential,
            conn_str=PROJECT_CONNECTION_STRING,
        )
        print("Connected to Azure AI Project")
        return project_client
    except Exception as e:
        print(f"Error connecting to Azure AI Project: {e}")
        
        # Try connecting without credential as fallback
        try:
            print("Trying to connect without credential parameter...")
            project_client = AIProjectClient.from_connection_string(
                conn_str=PROJECT_CONNECTION_STRING
            )
            print("Connected to Azure AI Project (fallback method)")
            return project_client
        except Exception as e2:
            print(f"Fallback connection also failed: {e2}")
            raise

def get_bing_grounding_tool(project_client):
    """Retrieve Bing Search connection and create a Bing grounding tool."""
    try:
        # Get the Bing connection
        bing_connection = project_client.connections.get(
            connection_name=BING_CONNECTION_NAME
        )
        conn_id = bing_connection.id
        print(f"Retrieved Bing connection ID: {conn_id}")
        
        # Initialize the Bing grounding tool
        bing_tool = BingGroundingTool(connection_id=conn_id)
        return bing_tool
    except Exception as e:
        print(f"Error setting up Bing grounding tool: {e}")
        raise

def extract_urls_from_text(text):
    """Extract URLs from text using regex."""
    url_pattern = r'https?://[^\s"\'\)\]}>]+'
    urls = re.findall(url_pattern, text)
    return urls

def create_grounded_agent(project_client, bing_tool):
    """Create an agent with Bing Search Grounding capability."""
    try:
        # Create agent with Bing grounding tool
        agent = project_client.agents.create_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="apicus-pricing-research-agent",
            instructions=(
                "You are a helpful assistant that specializes in researching software pricing information. "
                "When asked about pricing for specific software or apps, use Bing Search to find the EXACT URLs "
                "of official pricing pages. I need you to locate and provide direct links to the official "
                "pricing pages. Focus on finding the exact URLs rather than summarizing the pricing information."
                "Always include citations to your sources with the exact URLs."
            ),
            tools=bing_tool.definitions,
            headers={"x-ms-enable-preview": "true"}  # Enable preview features
        )
        print(f"Created Bing-grounded agent with ID: {agent.id}")
        return agent
    except Exception as e:
        print(f"Error creating Bing-grounded agent: {e}")
        raise

def create_pricing_extraction_tool():
    """
    Create a function calling tool definition based on the pricing schema.
    
    Returns:
        dict: Function tool definition for the Azure OpenAI API
    """
    # Use pricing schema to create the function tool
    # Convert simplified schema for function calling
    pricing_extraction_tool = {
        "type": "function",
        "function": {
            "name": "extract_pricing_data",
            "description": "Extract structured pricing data according to Apicus schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "description": "Unique identifier for the app"},
                    "app_name": {"type": "string", "description": "Display name of the app"},
                    "app_slug": {"type": "string", "description": "URL-friendly version of the app name"},
                    "pricing_url": {"type": "string", "description": "URL of the pricing page that was scraped"},
                    "all_pricing_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "All discovered pricing page URLs in order of confidence"
                    },
                    "price_model_type": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["subscription", "usage_based", "token_based", "free_tier", "one_time", "hybrid", "custom", "quote_based"]
                        },
                        "description": "The pricing model types used by this app"
                    },
                    "has_free_tier": {"type": "boolean", "description": "Whether the app offers a free tier"},
                    "has_free_trial": {"type": "boolean", "description": "Whether the app offers a free trial"},
                    "free_trial_period_days": {"type": ["integer", "null"], "description": "Duration of free trial in days, if available"},
                    "currency": {"type": "string", "description": "Primary currency used for pricing (e.g., USD, EUR)"},
                    "is_pricing_public": {"type": "boolean", "description": "Whether detailed pricing information is publicly available"},
                    "pricing_page_accessible": {"type": "boolean", "description": "Whether the pricing page was publicly accessible for scraping"},
                    "pricing_notes": {"type": "string", "description": "Additional notes or context on pricing structure or limitations"},
                    "pricing_tiers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tier_name": {"type": "string", "description": "Name of the pricing tier (e.g., Basic, Pro, Enterprise)"},
                                "tier_description": {"type": "string", "description": "Description of this tier"},
                                "monthly_price": {"type": "number", "description": "Price per month in the specified currency"},
                                "annual_price": {"type": "number", "description": "Price per year in the specified currency (may offer discount)"},
                                "annual_discount_percentage": {"type": "number", "description": "Percentage discount for annual billing vs monthly"},
                                "setup_fee": {"type": "number", "description": "One-time setup fee, if applicable"},
                                "features": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of features included in this tier"
                                },
                                "limits": {
                                    "type": "object",
                                    "properties": {
                                        "users": {"type": ["integer", "string"], "description": "Maximum number of users allowed or 'unlimited'"},
                                        "storage": {"type": "string", "description": "Storage limit (e.g., '10GB') or 'unlimited'"},
                                        "operations": {"type": ["integer", "string"], "description": "Number of operations included or 'unlimited'"},
                                        "api_calls": {"type": ["integer", "string"], "description": "Number of API calls included or 'unlimited'"},
                                        "custom_limits": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string", "description": "Name of the limit"},
                                                    "value": {"type": ["string", "number", "boolean"], "description": "Value of the limit"}
                                                }
                                            },
                                            "description": "Custom limits specific to this application"
                                        }
                                    },
                                    "description": "Usage limits for this tier"
                                }
                            },
                            "required": ["tier_name"]
                        },
                        "description": "Array of pricing tiers offered by the app"
                    },
                    "usage_based_pricing": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric_name": {"type": "string", "description": "Name of the usage metric (e.g., API calls, tokens, storage)"},
                                "unit": {"type": "string", "description": "Unit of measurement (e.g., 'per call', 'per 1K tokens')"},
                                "base_price": {"type": "number", "description": "Base price for this usage metric"},
                                "tiers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "min": {"type": "number", "description": "Minimum usage for this tier"},
                                            "max": {"type": ["number", "string"], "description": "Maximum usage for this tier or 'unlimited'"},
                                            "price": {"type": "number", "description": "Price per unit at this tier"}
                                        }
                                    },
                                    "description": "Volume-based pricing tiers, if applicable"
                                }
                            },
                            "required": ["metric_name"]
                        },
                        "description": "Usage-based pricing details, particularly relevant for AI services"
                    },
                    "ai_specific_pricing": {
                        "type": "object",
                        "properties": {
                            "has_token_based_pricing": {"type": "boolean", "description": "Whether the app uses token-based pricing (common for LLMs)"},
                            "input_token_price": {"type": "number", "description": "Price per input token"},
                            "output_token_price": {"type": "number", "description": "Price per output token"},
                            "models_pricing": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "model_name": {"type": "string", "description": "Name of AI model"},
                                        "input_price": {"type": "number", "description": "Input price for this model"},
                                        "output_price": {"type": "number", "description": "Output price for this model"},
                                        "unit": {"type": "string", "description": "Pricing unit (e.g., 'per 1K tokens')"}
                                    }
                                },
                                "description": "Pricing for different AI models offered"
                            }
                        },
                        "description": "AI-specific pricing details"
                    },
                    "promotional_offers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "offer_name": {"type": "string", "description": "Name of promotional offer"},
                                "offer_description": {"type": "string", "description": "Description of the offer"},
                                "discount_percentage": {"type": "number", "description": "Percentage discount offered"},
                                "offer_url": {"type": "string", "description": "URL pointing directly to promotional offer details"}
                            }
                        },
                        "description": "Current promotional offers or discounts"
                    }
                },
                "required": [
                    "app_id", "app_name", "app_slug", "pricing_url",
                    "price_model_type", "has_free_tier", "has_free_trial",
                    "currency", "is_pricing_public", "pricing_page_accessible"
                ]
            }
        }
    }
    
    return pricing_extraction_tool

def init_openai_client():
    """
    Initialize the Azure OpenAI client while avoiding the 'proxies' parameter issue.
    
    This is a wrapper function that properly handles client initialization
    while working around environment issues that might inject proxy settings.
    
    Returns:
        AzureOpenAI: An initialized Azure OpenAI client
    """
    from openai import AzureOpenAI
    import os
    import httpx
    
    # Store original environment variables
    original_env = os.environ.copy()
    
    try:
        # Temporarily remove any proxy settings from the environment
        for var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY'):
            if var in os.environ:
                del os.environ[var]
        
        # Initialize with minimal parameters to avoid proxies issue
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        
        return client
    
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {str(e)}")
        
        # Try an alternative method
        try:
            print("Trying alternative client initialization...")
            # Create a custom httpx client with no proxy settings
            http_client = httpx.Client(verify=True, follow_redirects=True)
            
            # Pass this client directly to avoid proxy settings
            client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                http_client=http_client
            )
            
            return client
        except Exception as e2:
            print(f"Alternative initialization also failed: {str(e2)}")
            raise
    
    finally:
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(original_env)

def analyze_pricing_with_openai(content, app_name, app_slug, is_pricing_public=True, is_page_accessible=True):
    """
    Use Azure OpenAI to analyze pricing page content and extract structured data
    using Pydantic models for structured outputs.
    
    Args:
        content (str): Text content from the pricing page
        app_name (str): Name of the app
        app_slug (str): URL-friendly version of app name
        is_pricing_public (bool): Whether the pricing appears to be publicly available
        is_page_accessible (bool): Whether the pricing page was publicly accessible
    
    Returns:
        dict: Structured pricing data based on our schema
    """
    if not content or len(content.strip()) < 50:
        print(f"Insufficient content for {app_name} to analyze")
        return None
    
    try:
        # Use our custom initialization function
        client = init_openai_client()
        
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
        max_content_length = 24000  # Reduced from 32k to ensure we don't exceed limits
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
                    
                    # Try to initialize a new client for this retry
                    client = init_openai_client()
                    
                    # Simplify the prompt for the next retry
                    if retry == max_retries - 2:  # Last retry
                        print("Simplifying prompt for final retry...")
                        system_prompt = f"""Extract basic pricing information for '{app_name}'. 
                        Focus on core pricing details: pricing tiers, prices, and free tier availability."""
                        
                        # Reduce content length further
                        max_content_length = max_content_length // 2
                        user_prompt = f"Extract pricing information for {app_name}:\n\n{content[:max_content_length]}"
                else:
                    # If we've exhausted all retries, create minimal pricing data
                    print(f"Structured output failed after {max_retries} attempts, creating minimal data")
                    return create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible)
        
        # If we've reached here, all retries failed
        return create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible)
        
    except Exception as e:
        print(f"Error in pricing analysis: {e}")
        return create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible)

def create_fallback_pricing_data(app_name, app_slug, is_pricing_public, is_page_accessible):
    """
    Create a minimal pricing data document when extraction fails.
    
    Args:
        app_name (str): Name of the app
        app_slug (str): URL-friendly version of app name
        is_pricing_public (bool): Whether the pricing appears to be publicly available
        is_page_accessible (bool): Whether the pricing page was publicly accessible
        
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

def search_pricing_with_bing(project_client, agent_id, app_name, app_slug=None):
    """
    Search for pricing information using Bing grounding.
    
    Args:
        project_client: Azure AI Project client
        agent_id: ID of the agent with Bing grounding
        app_name: Name of the app to search for
        app_slug: Optional URL-friendly version of app name
    
    Returns:
        dict: Search results with citations
    """
    try:
        # Create a thread for this conversation
        thread = project_client.agents.create_thread()
        thread_id = thread.id
        print(f"Created conversation thread with ID: {thread_id}")
        
        # Construct the search query
        query = f"Find the official pricing page URL for {app_name}"
        if app_slug:
            query += f" ({app_slug}.com)"
        query += ". Only provide the exact URL to the pricing page, not a summary of pricing information."
        
        # Add the user message (query) to the thread
        user_message = project_client.agents.create_message(
            thread_id=thread_id,
            role="user",
            content=query
        )
        print(f"Added user query to thread: {query}")
        
        # Run the agent to process the query
        print("Processing query with Bing-grounded agent...")
        run = project_client.agents.create_and_process_run(
            thread_id=thread_id, 
            agent_id=agent_id
        )
        
        # Check run status
        if run.status == "failed":
            print(f"Run failed with error: {run.last_error}")
            return {
                "error": f"Search failed: {run.last_error}",
                "app_name": app_name,
                "app_slug": app_slug,
                "timestamp": time.time()
            }
        
        # For debugging: retrieve run steps to see the Bing search query
        try:
            steps = project_client.agents.list_run_steps(run_id=run.id, thread_id=thread_id)
            for step in steps.data:
                if hasattr(step, 'tool_calls') and step.tool_calls:
                    for tool_call in step.tool_calls:
                        if tool_call.type == "bing_grounding":
                            print(f"Bing search query: {tool_call.bing_grounding.input}")
        except Exception as e:
            print(f"Could not retrieve run steps: {e}")
        
        # Get the assistant's response
        response_msg = project_client.agents.list_messages(thread_id=thread_id).get_last_message_by_role("assistant")
        
        if not response_msg:
            return {
                "error": "No response received from agent",
                "app_name": app_name,
                "app_slug": app_slug,
                "timestamp": time.time()
            }
        
        # Extract the response text and citations
        result = {
            "app_name": app_name,
            "app_slug": app_slug,
            "search_term": query,
            "timestamp": time.time(),
            "response_text": [],
            "citations": [],
            "extracted_content": []
        }
        
        # Extract text content
        response_text = ""
        for text_part in response_msg.text_messages:
            text = text_part.text.value
            response_text += text
            result["response_text"].append(text)
        
        # Extract URLs from the response text
        extracted_urls = extract_urls_from_text(response_text)
        
        # Extract citations
        citation_urls = []
        for annotation in response_msg.url_citation_annotations:
            citation = {
                "title": annotation.url_citation.title,
                "url": annotation.url_citation.url
            }
            result["citations"].append(citation)
            citation_urls.append(annotation.url_citation.url)
        
        # Combine unique URLs from both text extraction and citations
        all_urls = set(extracted_urls + citation_urls)
        pricing_urls = [url for url in all_urls if "/pric" in url.lower() or "/plan" in url.lower()]
        
        # If no specific pricing URLs found, use all URLs
        urls_to_extract = pricing_urls if pricing_urls else list(all_urls)
        
        # Limit to top 3 URLs
        urls_to_extract = urls_to_extract[:3]
        
        # Extract content from each URL using Jina.ai
        for url in urls_to_extract:
            print(f"Extracting content from URL: {url}")
            content, is_accessible, error = fetch_content_with_jina(url)
            
            content_result = {
                "url": url,
                "is_accessible": is_accessible,
                "content": content if content else None,
                "error": error
            }
            
            result["extracted_content"].append(content_result)
            
            if content:
                print(f"Successfully extracted content from: {url} ({len(content)} characters)")
                
                # Check if pricing is public
                is_pricing_public = check_pricing_is_public(content)
                
                # Analyze content with OpenAI using o4 model
                pricing_data = analyze_pricing_with_openai(content, app_name, app_slug, is_pricing_public, is_accessible)
                
                if pricing_data:
                    content_result["pricing_data"] = pricing_data
                    content_result["pricing_analyzed"] = True
                    print(f"Successfully extracted structured pricing data for {app_name}")
                else:
                    content_result["pricing_analyzed"] = False
                    print(f"Failed to extract structured pricing data for {app_name}")
            else:
                print(f"Failed to extract content from: {url}")
        
        # Clean up resources
        project_client.agents.delete_thread(thread_id)
        
        return result
    
    except Exception as e:
        print(f"Error in Bing grounding search: {e}")
        return {
            "error": f"Error: {str(e)}",
            "app_name": app_name,
            "app_slug": app_slug,
            "timestamp": time.time()
        }

def save_results_to_json(result, app_name, app_slug=None):
    """
    Save the search and extraction results to organized JSON files.
    
    Args:
        result (dict): The search and extraction results
        app_name (str): The name of the app
        app_slug (str, optional): The app slug
    
    Returns:
        str: Path to the results directory
    """
    import os
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    base_dir = "pricing_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create app-specific directory
    app_dir_name = app_slug.lower() if app_slug else app_name.lower().replace(" ", "_")
    app_dir = os.path.join(base_dir, app_dir_name)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(app_dir, timestamp)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Save the overall results
    overview_file = os.path.join(run_dir, "search_overview.json")
    overview_data = {
        "app_name": app_name,
        "app_slug": app_slug,
        "timestamp": datetime.now().isoformat(),
        "search_term": result.get("search_term"),
        "response_text": result.get("response_text"),
        "citations": result.get("citations"),
        "extracted_urls": [item.get("url") for item in result.get("extracted_content", [])]
    }
    
    with open(overview_file, "w", encoding="utf-8") as f:
        json.dump(overview_data, f, indent=2)
    
    # Save each URL's extracted content and pricing data
    for i, content_item in enumerate(result.get("extracted_content", [])):
        url = content_item.get("url", f"unknown_url_{i}")
        url_filename = f"url_{i+1}_{url.split('/')[-1].split('?')[0][:30]}.json"
        url_file = os.path.join(run_dir, url_filename)
        
        url_data = {
            "url": url,
            "is_accessible": content_item.get("is_accessible", False),
            "error": content_item.get("error"),
            "content_length": len(content_item.get("content", "")) if content_item.get("content") else 0,
            "pricing_analyzed": content_item.get("pricing_analyzed", False),
        }
        
        # Include content preview (optional - can be large)
        content = content_item.get("content")
        if content:
            url_data["content_preview"] = content[:1000] + "..." if len(content) > 1000 else content
        
        # Include pricing data if available
        if "pricing_data" in content_item:
            url_data["pricing_data"] = content_item["pricing_data"]
            
            # Also save pricing data to its own file for easy access
            pricing_file = os.path.join(run_dir, f"pricing_data_{i+1}.json")
            with open(pricing_file, "w", encoding="utf-8") as f:
                json.dump(content_item["pricing_data"], f, indent=2)
        
        with open(url_file, "w", encoding="utf-8") as f:
            json.dump(url_data, f, indent=2)
    
    print(f"Results saved to directory: {run_dir}")
    return run_dir

def main(app_name, app_slug=None):
    """Main function to run a Bing-grounded search for an app's pricing."""
    try:
        # Print configuration info
        print("\nConfiguration:")
        print(f"- Bing Grounding Agent model: {MODEL_DEPLOYMENT_NAME}")
        print(f"- Pricing Analysis model: {AZURE_DEPLOYMENT}")
        print(f"- Bing Connection: {BING_CONNECTION_NAME}")
        print(f"- API Version: {API_VERSION}")
        
        # Initialize project client
        project_client = initialize_project_client()
        
        # Get Bing grounding tool
        bing_tool = get_bing_grounding_tool(project_client)
        
        # Create agent with Bing grounding
        agent = create_grounded_agent(project_client, bing_tool)
        
        # Search for pricing with Bing
        result = search_pricing_with_bing(project_client, agent.id, app_name, app_slug)
        
        # Save results to JSON files
        results_dir = save_results_to_json(result, app_name, app_slug)
        result["results_directory"] = results_dir
        
        # Clean up the agent
        project_client.agents.delete_agent(agent.id)
        
        return result
    
    except Exception as e:
        print(f"Error in Bing grounding pipeline: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        app_name = sys.argv[1]
        app_slug = sys.argv[2] if len(sys.argv) > 2 else None
        result = main(app_name, app_slug)
        print("\nSearch Result:")
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python bing_grounding_search.py \"App Name\" [app-slug]") 
#!/usr/bin/env python3
"""
Test script to verify the pricing pipeline is working correctly
"""

import os
import json
import sys
from dotenv import load_dotenv
import logging
from scrape_w_jina_ai import fetch_content_with_jina

# Load environment variables
load_dotenv()

# Disable any proxy settings that might be in the environment
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']
if 'HTTP_PROXY' in os.environ:
    del os.environ['HTTP_PROXY']
if 'HTTPS_PROXY' in os.environ:
    del os.environ['HTTPS_PROXY']

def test_openai_client():
    """Test if we can create an OpenAI client without the proxies error"""
    try:
        print("Testing basic OpenAI client initialization...")
        from openai import AzureOpenAI
        
        # Get configuration from environment
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
        
        print(f"Using endpoint: {endpoint}")
        print(f"Using API version: {api_version}")
        
        # Create a minimal client with no extra parameters
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        print("Successfully created OpenAI client!")
        
        # Try a simple API call
        print("Testing API call...")
        response = client.chat.completions.create(
            model=os.getenv("AZURE_o3_DEPLOYMENT", "o3-mini-apicus"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, what's 2+2?"}
            ],
            max_tokens=10
        )
        
        print(f"API response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_content_extraction():
    """Test extracting content from a URL using Jina.ai"""
    url = "https://about.gitlab.com/pricing/"
    print(f"Testing content extraction from {url}...")
    
    content, is_accessible, error = fetch_content_with_jina(url)
    
    if is_accessible:
        print(f"Successfully extracted {len(content)} characters")
        print("First 300 characters of content:")
        print(content[:300])
        return True
    else:
        print(f"Failed to extract content: {error}")
        return False

if __name__ == "__main__":
    # Enable verbose logging
    logging.basicConfig(level=logging.DEBUG)
    
    # First test content extraction with Jina
    print("\n=== TESTING JINA CONTENT EXTRACTION ===")
    jina_success = test_content_extraction()
    
    # Then test OpenAI client
    print("\n=== TESTING OPENAI CLIENT ===")
    openai_success = test_openai_client()
    
    # Report overall success
    if jina_success and openai_success:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ One or more tests failed.")
        sys.exit(1) 
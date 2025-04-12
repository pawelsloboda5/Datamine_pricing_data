# Apicus App Pricing Scraper

This project automatically discovers pricing pages for software applications, extracts structured pricing information, and stores it in MongoDB.

## Overview

The pipeline consists of three main phases:

1. **Discovery**: Find pricing pages for apps using Jina.ai's web search
2. **Extraction**: Extract structured pricing information from discovered pages using Azure OpenAI
3. **Storage**: Store pricing data in MongoDB with vector embeddings for semantic search

## Features

- Automatically discovers official pricing pages for apps
- Extracts detailed pricing information including tiers, features, and limits
- Supports various pricing models (subscription, usage-based, token-based, etc.)
- Handles AI-specific pricing details for AI platforms
- Detects whether pricing is public or requires contacting sales
- Flags accessibility issues with pricing pages
- Creates vector embeddings for semantic search
- Stores raw pricing text for verification or reprocessing
- Enhanced Bing Search grounding for more accurate pricing page discovery
- Robust JSON repair for handling unterminated strings and malformed responses
- Quality rating system to select the best pricing data source
- Structured outputs using Pydantic models for type-safe, validated extractions

## Prerequisites

- Python 3.7+
- MongoDB instance
- Azure OpenAI API access
- Azure AI Agent Service with Bing Search Grounding connection (for enhanced pipeline)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:

```
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB_NAME=your_database_name

AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-10-21

AZURE_o3_DEPLOYMENT=your_o3_model_deployment_name
AZURE_4o_MINI_DEPLOYMENT=your_4omini_model_deployment_name
AZURE_o4_DEPLOYMENT=your_o4_model_deployment_name
AZURE_TEXT_EMBEDDING_3_SMALL_DEPLOYMENT=your_embedding_model_deployment_name

# For enhanced pipeline with Bing Search Grounding
PROJECT_CONNECTION_STRING=your_azure_ai_project_connection_string
BING_CONNECTION_NAME=your_bing_connection_name
```

## Usage

The project provides a main command-line interface for running different parts of the pipeline:

### Discover pricing pages

```bash
python main.py discover --limit 20
```

This will search for pricing pages for up to 20 apps and store the results in the `apicus-apps-prices-discovery` collection.

### Extract pricing data

```bash
python main.py extract --limit 10
```

This will process up to 10 apps with discovered pricing pages, extract their pricing data, and store it in the `apicus-apps-prices` collection.

### Run the full pipeline

```bash
python main.py pipeline --discover-limit 20 --extract-limit 10
```

This will run the complete pipeline: first discovering pricing pages for up to 20 apps, then extracting pricing data for up to 10 of them.

### Run enhanced pipeline with Bing Search grounding (recommended)

```bash
python main.py enhanced-pipeline --limit 10
```

This runs the enhanced pipeline that uses Azure AI Agent Service with Bing Search grounding to find the most up-to-date pricing pages, extracts and validates pricing data, and selects the best result based on confidence scoring. Results are saved both to files and MongoDB.

For a single app:

```bash
python main.py enhanced-pipeline --app-name "Slack" --app-slug "slack"
```

### Use Bing Search directly

```bash
python main.py bing --app-name "Notion" --app-slug "notion"
```

This will search for pricing pages using Azure Bing Search grounding and display results.

## Project Structure

- `main.py`: Main script that coordinates the entire pipeline
- `scrape_w_jina_ai.py`: Handles discovery of pricing pages using Jina.ai
- `extract_pricing_data.py`: Extracts structured pricing data from web pages
- `bing_grounding_search.py`: Uses Azure Bing Search grounding for more accurate results
- `pricing_schema.py`: Defines the JSON schema for pricing data
- `pricing_models.py`: Defines Pydantic models for structured outputs

## Structured Outputs

The application now uses Azure OpenAI's structured outputs feature with Pydantic models to ensure strict schema adherence. This approach offers several advantages:

1. **Strict Schema Validation**: Schema is enforced by the model itself, not just suggested
2. **Type Safety**: Proper typing of all fields (strings, numbers, booleans, etc.)
3. **Reduced Errors**: Eliminates JSON parsing errors and malformed outputs
4. **Simpler Code**: Cleaner code that doesn't need complex error handling for malformed JSON

Example of the improved approach:

```python
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# Extract structured data with a Pydantic model
completion = client.beta.chat.completions.parse(
    model=AZURE_DEPLOYMENT,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format=PricingData,  # Pydantic model that defines the schema
    temperature=0.1
)

# Access the parsed data directly
pricing_data = completion.choices[0].message.parsed
```

This approach requires using Azure OpenAI API version 2024-10-21 or newer and models that support the structured outputs feature (such as o4, o3-mini, and gpt-4o).

## Data Schema

The pricing data follows this enhanced schema (see `pricing_schema.py` for details):

```json
{
  "app_id": "unique_app_identifier",
  "app_name": "App Name",
  "app_slug": "app-name",
  "pricing_url": "https://example.com/pricing",
  "source_url": "https://example.com/pricing", // The exact URL data was extracted from
  "all_pricing_urls": ["https://example.com/pricing", "https://example.com/plans"],
  "pricing_captured_at": {"$date": "2023-08-15T12:00:00Z"},
  "last_updated": {"$date": "2023-08-15T12:00:00Z"},
  "is_pricing_public": true,
  "pricing_page_accessible": true,
  "schema_validated": true, // Whether the data passed schema validation
  "confidence_score": 85, // Confidence score (0-100) for data quality
  "price_model_type": ["subscription", "free_tier"],
  "has_free_tier": true,
  "has_free_trial": true,
  "free_trial_period_days": 14,
  "currency": "USD",
  "pricing_notes": "Special discounts available for educational institutions.",
  "pricing_tiers": [
    {
      "tier_name": "Basic",
      "monthly_price": 10,
      "annual_price": 100,
      "annual_discount_percentage": 16.7,
      "features": ["Feature 1", "Feature 2"],
      "limits": {
        "users": 5,
        "storage": "10GB",
        "api_calls": "unlimited"
      }
    },
    {
      "tier_name": "Pro",
      "monthly_price": 20,
      "annual_price": 200,
      "features": ["Feature 1", "Feature 2", "Feature 3"]
    }
  ],
  "usage_based_pricing": [
    {
      "metric_name": "API Calls",
      "unit": "per 1000 calls",
      "base_price": 5,
      "tiers": [
        {
          "min": 0,
          "max": 10000,
          "price": 5
        },
        {
          "min": 10001,
          "max": "unlimited",
          "price": 3
        }
      ]
    }
  ],
  "promotional_offers": [
    {
      "offer_name": "Summer Sale",
      "offer_description": "20% off all plans",
      "discount_percentage": 20,
      "offer_url": "https://example.com/summer-sale"
    }
  ],
  "extraction_timestamp": "2023-08-15T12:00:00Z",
  "json_repaired": false // Whether JSON repair was needed during extraction
}
```

## Enhanced Pipeline Process

The enhanced pipeline performs the following steps:

1. **Retrieval**: Get app information from MongoDB or use provided app name/slug
2. **Discovery**: Use Azure AI Agent Service with Bing Search grounding to find official pricing pages
3. **Content Extraction**: Extract clean text content from each URL
4. **Analysis**: Use Azure OpenAI to analyze content and extract structured pricing data
5. **Validation**: Validate data against the schema to ensure completeness
6. **Confidence Scoring**: Calculate a confidence score based on data completeness
7. **Selection**: Select the best pricing data based on confidence scores
8. **Storage**: Store data in MongoDB and save JSON files in app-specific directories

## Error Handling and Repair

The system includes robust error handling and repair capabilities:

- **JSON Repair**: Fixes common JSON issues like unterminated strings, missing brackets, or trailing commas
- **Error Logging**: Logs full error responses to help diagnose extraction issues
- **Fallback Options**: Creates minimal valid documents when full extraction fails
- **Multiple URL Attempts**: Tries multiple pricing URLs in order of confidence

## MongoDB Collections

- `apicus-processed-apps`: Source collection with app metadata
- `apicus-apps-prices-discovery`: Intermediate collection with discovered pricing page URLs
- `apicus-apps-prices`: Final collection with structured pricing data

## License

This project is proprietary and confidential. 
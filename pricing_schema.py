"""
Schema definition for app pricing data.
This module contains the JSON schema for app pricing data and helper functions.
"""

PRICING_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Apicus App Pricing Schema",
    "type": "object",
    "properties": {
        "_id": {
            "type": "object",
            "properties": {
                "$oid": {
                    "type": "string",
                    "description": "MongoDB ObjectId"
                }
            },
            "required": ["$oid"]
        },
        "app_id": {
            "type": "string",
            "description": "Unique identifier for the app, matching the app_id in apicus-processed-apps"
        },
        "app_name": {
            "type": "string",
            "description": "Display name of the app"
        },
        "app_slug": {
            "type": "string",
            "description": "URL-friendly version of the app name"
        },
        "pricing_url": {
            "type": "string",
            "description": "URL of the pricing page that was scraped"
        },
        "all_pricing_urls": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "All discovered pricing page URLs in order of confidence"
        },
        "pricing_captured_at": {
            "type": "object",
            "properties": {
                "$date": {
                    "type": "string",
                    "format": "date-time",
                    "description": "When the pricing data was captured"
                }
            },
            "required": ["$date"]
        },
        "last_updated": {
            "type": "object",
            "properties": {
                "$date": {
                    "type": "string",
                    "format": "date-time",
                    "description": "When the pricing data was last updated or verified"
                }
            },
            "required": ["$date"]
        },
        "price_model_type": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["subscription", "usage_based", "token_based", "free_tier", "one_time", "hybrid", "custom", "quote_based"]
            },
            "description": "The pricing model types used by this app"
        },
        "has_free_tier": {
            "type": "boolean",
            "description": "Whether the app offers a free tier"
        },
        "has_free_trial": {
            "type": "boolean",
            "description": "Whether the app offers a free trial"
        },
        "free_trial_period_days": {
            "type": ["integer", "null"],
            "description": "Duration of free trial in days, if available"
        },
        "currency": {
            "type": "string",
            "description": "Primary currency used for pricing (e.g., USD, EUR)"
        },
        "is_pricing_public": {
            "type": "boolean",
            "description": "Whether detailed pricing information is publicly available"
        },
        "pricing_page_accessible": {
            "type": "boolean",
            "description": "Whether the pricing page was publicly accessible for scraping"
        },
        "pricing_notes": {
            "type": "string",
            "description": "Additional notes or context on pricing structure or limitations"
        },
        "pricing_tiers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tier_name": {
                        "type": "string",
                        "description": "Name of the pricing tier (e.g., Basic, Pro, Enterprise)"
                    },
                    "tier_description": {
                        "type": "string",
                        "description": "Description of this tier"
                    },
                    "monthly_price": {
                        "type": "number",
                        "description": "Price per month in the specified currency"
                    },
                    "annual_price": {
                        "type": "number",
                        "description": "Price per year in the specified currency (may offer discount)"
                    },
                    "annual_discount_percentage": {
                        "type": "number",
                        "description": "Percentage discount for annual billing vs monthly"
                    },
                    "setup_fee": {
                        "type": "number",
                        "description": "One-time setup fee, if applicable"
                    },
                    "features": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of features included in this tier"
                    },
                    "limits": {
                        "type": "object",
                        "properties": {
                            "users": {
                                "type": ["integer", "string"],
                                "description": "Maximum number of users allowed or 'unlimited'"
                            },
                            "storage": {
                                "type": "string",
                                "description": "Storage limit (e.g., '10GB') or 'unlimited'"
                            },
                            "operations": {
                                "type": ["integer", "string"],
                                "description": "Number of operations included or 'unlimited'"
                            },
                            "api_calls": {
                                "type": ["integer", "string"],
                                "description": "Number of API calls included or 'unlimited'"
                            },
                            "custom_limits": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the limit"
                                        },
                                        "value": {
                                            "type": ["string", "number", "boolean"],
                                            "description": "Value of the limit"
                                        }
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
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the usage metric (e.g., API calls, tokens, storage)"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g., 'per call', 'per 1K tokens')"
                    },
                    "base_price": {
                        "type": "number",
                        "description": "Base price for this usage metric"
                    },
                    "tiers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "min": {
                                    "type": "number",
                                    "description": "Minimum usage for this tier"
                                },
                                "max": {
                                    "type": ["number", "string"],
                                    "description": "Maximum usage for this tier or 'unlimited'"
                                },
                                "price": {
                                    "type": "number",
                                    "description": "Price per unit at this tier"
                                }
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
                "has_token_based_pricing": {
                    "type": "boolean",
                    "description": "Whether the app uses token-based pricing (common for LLMs)"
                },
                "input_token_price": {
                    "type": "number",
                    "description": "Price per input token"
                },
                "output_token_price": {
                    "type": "number",
                    "description": "Price per output token"
                },
                "models_pricing": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of AI model"
                            },
                            "input_price": {
                                "type": "number",
                                "description": "Input price for this model"
                            },
                            "output_price": {
                                "type": "number",
                                "description": "Output price for this model"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Pricing unit (e.g., 'per 1K tokens')"
                            }
                        }
                    },
                    "description": "Pricing for different AI models offered"
                },
                "has_inference_pricing": {
                    "type": "boolean",
                    "description": "Whether the app has special pricing for inference"
                },
                "has_fine_tuning_pricing": {
                    "type": "boolean",
                    "description": "Whether the app has special pricing for fine-tuning"
                },
                "has_training_pricing": {
                    "type": "boolean",
                    "description": "Whether the app has special pricing for model training"
                }
            },
            "description": "AI-specific pricing details"
        },
        "promotional_offers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "offer_name": {
                        "type": "string",
                        "description": "Name of promotional offer"
                    },
                    "offer_description": {
                        "type": "string",
                        "description": "Description of the offer"
                    },
                    "discount_percentage": {
                        "type": "number",
                        "description": "Percentage discount offered"
                    },
                    "offer_url": {
                        "type": "string",
                        "description": "URL pointing directly to promotional offer details"
                    },
                    "valid_until": {
                        "type": "string",
                        "format": "date-time",
                        "description": "When the promotion expires"
                    }
                }
            },
            "description": "Current promotional offers or discounts"
        },
        "additional_fees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fee_name": {
                        "type": "string",
                        "description": "Name of additional fee"
                    },
                    "fee_amount": {
                        "type": "number",
                        "description": "Amount of the fee"
                    },
                    "fee_description": {
                        "type": "string",
                        "description": "Description of what the fee covers"
                    }
                }
            },
            "description": "Additional fees not included in base pricing"
        },
        "raw_pricing_text": {
            "type": "string",
            "description": "Raw text extracted from the pricing page for verification or reprocessing"
        },
        "embedding_vector": {
            "type": "array",
            "items": {
                "type": "number"
            },
            "description": "Vector embedding for semantic search"
        }
    },
    "required": [
        "app_id",
        "app_name",
        "app_slug",
        "pricing_url",
        "pricing_captured_at",
        "price_model_type",
        "has_free_tier",
        "has_free_trial",
        "currency",
        "is_pricing_public",
        "pricing_page_accessible"
    ]
}

# Extract pricing schema in format suitable for OpenAI prompt
def get_schema_for_prompt():
    """
    Returns a simplified version of the pricing schema suitable for inclusion in an OpenAI prompt.
    
    Returns:
        str: JSON schema as a string, formatted for inclusion in a prompt
    """
    # Create a simplified version without extensive descriptions
    prompt_schema = {
        "app_id": "string - Unique identifier for the app",
        "app_name": "string - Display name of the app",
        "app_slug": "string - URL-friendly version of the app name",
        "pricing_url": "string - URL of the pricing page that was scraped",
        "all_pricing_urls": "array of strings - All discovered pricing page URLs in order of confidence",
        "is_pricing_public": "boolean - Whether detailed pricing information is publicly available",
        "pricing_page_accessible": "boolean - Whether the pricing page was publicly accessible for scraping",
        "price_model_type": "array of strings - The pricing model types (subscription, usage_based, token_based, free_tier, one_time, hybrid, custom, quote_based)",
        "has_free_tier": "boolean - Whether the app offers a free tier",
        "has_free_trial": "boolean - Whether the app offers a free trial",
        "free_trial_period_days": "integer - Duration of free trial in days, if available",
        "currency": "string - Primary currency used for pricing (e.g., USD, EUR)",
        "pricing_notes": "string - Additional notes or context on pricing structure or limitations",
        "pricing_tiers": [
            {
                "tier_name": "string - Name of the pricing tier (e.g., Basic, Pro, Enterprise)",
                "tier_description": "string - Description of this tier",
                "monthly_price": "number - Price per month",
                "annual_price": "number - Price per year",
                "annual_discount_percentage": "number - Percentage discount for annual billing",
                "features": ["string - Feature descriptions"],
                "limits": {
                    "users": "integer or 'unlimited' - Maximum number of users allowed",
                    "storage": "string - Storage limit (e.g., '10GB') or 'unlimited'",
                    "operations": "integer or 'unlimited' - Number of operations included",
                    "api_calls": "integer or 'unlimited' - Number of API calls included",
                    "custom_limits": [
                        {
                            "name": "string - Name of the limit",
                            "value": "any - Value of the limit"
                        }
                    ]
                }
            }
        ],
        "usage_based_pricing": [
            {
                "metric_name": "string - Name of the usage metric",
                "unit": "string - Unit of measurement",
                "base_price": "number - Base price",
                "tiers": [
                    {
                        "min": "number - Minimum usage for this tier",
                        "max": "number or 'unlimited' - Maximum usage for this tier",
                        "price": "number - Price per unit at this tier"
                    }
                ]
            }
        ],
        "ai_specific_pricing": {
            "has_token_based_pricing": "boolean - Whether the app uses token-based pricing",
            "input_token_price": "number - Price per input token",
            "output_token_price": "number - Price per output token",
            "models_pricing": [
                {
                    "model_name": "string - Name of AI model",
                    "input_price": "number - Input price",
                    "output_price": "number - Output price",
                    "unit": "string - Pricing unit"
                }
            ]
        },
        "promotional_offers": [
            {
                "offer_name": "string - Name of promotional offer",
                "offer_description": "string - Description of the offer",
                "discount_percentage": "number - Percentage discount offered",
                "offer_url": "string - URL pointing directly to promotional offer details"
            }
        ]
    }
    
    import json
    return json.dumps(prompt_schema, indent=2)

# Example of creating an empty pricing document template
def create_empty_pricing_doc(app_id, app_name, app_slug, pricing_url):
    """
    Create an empty pricing document with required fields filled in.
    
    Args:
        app_id (str): Unique ID of the app
        app_name (str): Name of the app
        app_slug (str): URL-friendly app name
        pricing_url (str): URL of the pricing page
    
    Returns:
        dict: Empty pricing document with required fields
    """
    from datetime import datetime
    
    current_time = datetime.now().isoformat()
    
    return {
        "app_id": app_id,
        "app_name": app_name,
        "app_slug": app_slug,
        "pricing_url": pricing_url,
        "all_pricing_urls": [pricing_url] if pricing_url else [],
        "pricing_captured_at": {"$date": current_time},
        "last_updated": {"$date": current_time},
        "price_model_type": [],
        "has_free_tier": False,
        "has_free_trial": False,
        "free_trial_period_days": None,
        "currency": "USD",  # Default to USD, will be updated if detected
        "is_pricing_public": True,  # Default to true, will be updated during extraction
        "pricing_page_accessible": True,  # Default to true, will be updated during extraction
        "pricing_notes": "",
        "pricing_tiers": [],
        "usage_based_pricing": [],
        "ai_specific_pricing": {},
        "promotional_offers": [],
        "additional_fees": []
    } 
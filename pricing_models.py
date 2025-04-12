from typing import List, Dict, Union, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class PriceModelType(str, Enum):
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    TOKEN_BASED = "token_based"
    FREE_TIER = "free_tier"
    ONE_TIME = "one_time"
    HYBRID = "hybrid"
    CUSTOM = "custom"
    QUOTE_BASED = "quote_based"


class CustomLimit(BaseModel):
    name: str = Field(description="Name of the limit")
    value: Union[str, int, float, bool] = Field(description="Value of the limit")


class Limits(BaseModel):
    users: Optional[Union[int, str]] = Field(None, description="Maximum number of users allowed or 'unlimited'")
    storage: Optional[str] = Field(None, description="Storage limit (e.g., '10GB') or 'unlimited'")
    operations: Optional[Union[int, str]] = Field(None, description="Number of operations included or 'unlimited'")
    api_calls: Optional[Union[int, str]] = Field(None, description="Number of API calls included or 'unlimited'")
    integrations: Optional[Union[int, str]] = Field(None, description="Number of integrations allowed or 'unlimited'")
    custom_limits: Optional[List[CustomLimit]] = Field(None, description="Custom limits specific to this application")


class PricingTier(BaseModel):
    tier_name: str = Field(description="Name of the pricing tier (e.g., Basic, Pro, Enterprise)")
    tier_description: Optional[str] = Field(None, description="Description of this tier")
    monthly_price: Optional[float] = Field(None, description="Price per month in the specified currency")
    annual_price: Optional[float] = Field(None, description="Price per year in the specified currency (may offer discount)")
    annual_discount_percentage: Optional[float] = Field(None, description="Percentage discount for annual billing vs monthly")
    setup_fee: Optional[float] = Field(None, description="One-time setup fee, if applicable")
    features: Optional[List[str]] = Field(None, description="List of features included in this tier")
    limits: Optional[Limits] = Field(None, description="Usage limits for this tier")


class UsageTier(BaseModel):
    min: Optional[float] = Field(None, description="Minimum usage for this tier")
    max: Optional[Union[float, str]] = Field(None, description="Maximum usage for this tier or 'unlimited'")
    price: Optional[float] = Field(None, description="Price per unit at this tier")


class UsageBasedPricing(BaseModel):
    metric_name: str = Field(description="Name of the usage metric (e.g., API calls, tokens, storage)")
    unit: Optional[str] = Field(None, description="Unit of measurement (e.g., 'per call', 'per 1K tokens')")
    base_price: Optional[float] = Field(None, description="Base price for this usage metric")
    tiers: Optional[List[UsageTier]] = Field(None, description="Volume-based pricing tiers, if applicable")


class ModelPricing(BaseModel):
    model_name: str = Field(description="Name of AI model")
    input_price: Optional[float] = Field(None, description="Input price for this model")
    output_price: Optional[float] = Field(None, description="Output price for this model")
    unit: Optional[str] = Field(None, description="Pricing unit (e.g., 'per 1K tokens')")


class AISpecificPricing(BaseModel):
    has_token_based_pricing: Optional[bool] = Field(None, description="Whether the app uses token-based pricing (common for LLMs)")
    input_token_price: Optional[float] = Field(None, description="Price per input token")
    output_token_price: Optional[float] = Field(None, description="Price per output token")
    models_pricing: Optional[List[ModelPricing]] = Field(None, description="Pricing for different AI models offered")
    has_inference_pricing: Optional[bool] = Field(None, description="Whether the app has special pricing for inference")
    has_fine_tuning_pricing: Optional[bool] = Field(None, description="Whether the app has special pricing for fine-tuning")
    has_training_pricing: Optional[bool] = Field(None, description="Whether the app has special pricing for model training")
    ai_addon_available: Optional[bool] = Field(None, description="Whether the app has an AI-specific addon available")


class PromotionalOffer(BaseModel):
    offer_name: str = Field(description="Name of promotional offer")
    offer_description: Optional[str] = Field(None, description="Description of the offer")
    discount_percentage: Optional[float] = Field(None, description="Percentage discount offered")
    offer_url: Optional[str] = Field(None, description="URL pointing directly to promotional offer details")
    valid_until: Optional[str] = Field(None, description="When the promotion expires")


class AdditionalFee(BaseModel):
    fee_name: str = Field(description="Name of additional fee")
    fee_amount: Optional[float] = Field(None, description="Amount of the fee")
    fee_description: Optional[str] = Field(None, description="Description of what the fee covers")


class PricingData(BaseModel):
    app_id: str = Field(description="Unique identifier for the app")
    app_name: str = Field(description="Display name of the app")
    app_slug: Optional[str] = Field(None, description="URL-friendly version of the app name")
    pricing_url: Optional[str] = Field(None, description="URL of the pricing page that was scraped")
    source_url: Optional[str] = Field(None, description="The exact URL data was extracted from") 
    all_pricing_urls: Optional[List[str]] = Field(None, description="All discovered pricing page URLs in order of confidence")
    price_model_type: List[PriceModelType] = Field(description="The pricing model types used by this app")
    has_free_tier: bool = Field(description="Whether the app offers a free tier")
    has_free_trial: bool = Field(description="Whether the app offers a free trial")
    free_trial_period_days: Optional[int] = Field(None, description="Duration of free trial in days, if available")
    currency: str = Field(description="Primary currency used for pricing (e.g., USD, EUR)")
    is_pricing_public: bool = Field(description="Whether detailed pricing information is publicly available")
    pricing_page_accessible: bool = Field(description="Whether the pricing page was publicly accessible for scraping")
    pricing_notes: Optional[str] = Field(None, description="Additional notes or context on pricing structure or limitations")
    pricing_tiers: Optional[List[PricingTier]] = Field(None, description="Array of pricing tiers offered by the app")
    usage_based_pricing: Optional[List[UsageBasedPricing]] = Field(None, description="Usage-based pricing details, particularly relevant for AI services")
    ai_specific_pricing: Optional[AISpecificPricing] = Field(None, description="AI-specific pricing details")
    promotional_offers: Optional[List[PromotionalOffer]] = Field(None, description="Current promotional offers or discounts")
    additional_fees: Optional[List[AdditionalFee]] = Field(None, description="Additional fees not included in base pricing")
    extraction_timestamp: Optional[str] = Field(None, description="When the pricing data was extracted")
    schema_validated: Optional[bool] = Field(None, description="Whether the data passed schema validation")
    confidence_score: Optional[int] = Field(None, description="Confidence score (0-100) for data quality")
    extraction_error: Optional[bool] = Field(None, description="Whether there was an error during extraction")
    json_repaired: Optional[bool] = Field(None, description="Whether JSON repair was needed during extraction") 
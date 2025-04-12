"""
Configuration settings for the application.
Loads environment variables and provides them as config values.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB settings
mongodb_uri = os.getenv("MONGODB_URI")
mongodb_db_name = os.getenv("MONGODB_DB_NAME", "apicus-db-data")

# Azure OpenAI settings
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

# Model deployments
openai_model = os.getenv("AZURE_o4_DEPLOYMENT", "apicus-gpt-4o")
embedding_model = os.getenv("AZURE_TEXT_EMBEDDING_3_SMALL_DEPLOYMENT", "text-embedding-3-small-apicus")

# Bing search settings
project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
bing_connection_name = os.getenv("BING_CONNECTION_NAME", "apicus-bing-connection") 
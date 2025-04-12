# Apicus Index and Vector Tracking

This document tracks the indexes, vector embeddings, and search configurations used in the Apicus platform.

## MongoDB Indexes

### apicus-processed-actions

| Index Name | Fields | Type | Description |
|------------|--------|------|-------------|
| _id_1 | _id | Default | Default MongoDB index on _id field |
| app_id_1 | app_id | Regular | For filtering actions by app |
| action_id_1 | action_id | Regular | For direct lookup of actions by ID |
| app_name_1 | app_name | Regular | For filtering actions by app name |
| title_1 | title | Regular | For text search on action title |
| type_1 | type | Regular | For filtering actions by type (write, search_or_write) |
| normalized_categories_1 | normalized_categories | Regular | For filtering actions by category |

### apicus-processed-triggers

| Index Name | Fields | Type | Description |
|------------|--------|------|-------------|
| _id_1 | _id | Default | Default MongoDB index on _id field |
| app_id_1 | app_id | Regular | For filtering triggers by app |
| trigger_id_1 | trigger_id | Regular | For direct lookup of triggers by ID |
| app_name_1 | app_name | Regular | For filtering triggers by app name |
| title_1 | title | Regular | For text search on trigger title |
| type_1 | type | Regular | For filtering triggers by type (instant, scheduled) |
| normalized_categories_1 | normalized_categories | Regular | For filtering triggers by category |

### apicus-processed-templates

| Index Name | Fields | Type | Description |
|------------|--------|------|-------------|
| _id_1 | _id | Default | Default MongoDB index on _id field |
| template_id_1 | template_id | Regular | For direct lookup of templates by ID |
| app_ids_1 | app_ids | Regular | For finding templates by app ID |
| step_count_1 | step_count | Regular | For filtering templates by step count |

### apicus-processed-apps

| Index Name | Fields | Type | Description |
|------------|--------|------|-------------|
| _id_1 | _id | Default | Default MongoDB index on _id field |
| app_id_1 | app_id | Regular | For direct lookup of apps by ID |
| name_1 | name | Regular | For text search on app name |
| slug_1 | slug | Regular | For direct lookup of apps by slug |
| normalized_categories_1 | normalized_categories | Regular | For filtering apps by category |

## Azure AI Search Configuration

### apicus-vector-search

- **Search Service Name**: apicus-ai-search
- **Location**: Central US
- **Pricing Tier**: Basic
- **Replicas**: 1
- **Partitions**: 1
- **Search Units**: 1

### Vector Search Settings

- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Parameters**:
  - m: 4 (Number of connections per layer)
  - efConstruction: 400 (Construction time/accuracy trade-off)
  - efSearch: 500 (Search time/accuracy trade-off)
  - metric: cosine (Similarity metric)

### Index Fields

| Field Name | Type | Search Configurations | Description |
|------------|------|----------------------|-------------|
| id | Edm.String | key, retrievable | Unique identifier |
| title | Edm.String | searchable, retrievable | Title for display |
| description | Edm.String | searchable, retrievable | Description text |
| app_id | Edm.String | filterable, retrievable | App ID for filtering |
| app_name | Edm.String | filterable, retrievable | App name for filtering |
| type | Edm.String | filterable, retrievable | Component type for filtering |
| categories | Collection(Edm.String) | filterable, retrievable | Categories for filtering |
| embedding | Collection(Edm.Single) | vector dimensions=1536, vector profile=vector-profile | Vector embedding array |
| document_type | Edm.String | filterable, retrievable | Type (actions, triggers) for filtering |
| normalized_fields | Collection(Edm.String) | filterable, retrievable | Field names for filtering |
| rich_description | Edm.String | searchable, retrievable | Rich text description |

## Vector Generation

- **Model**: text-embedding-3-large (Azure OpenAI)
- **Dimensions**: 3072
- **Embedding Text**: Combination of title, description, rich_description, and normalized_categories

### Process Steps

1. Create Azure AI Search service
2. Configure HNSW vector search algorithm
3. Define index schema with vector field
4. Extract documents from MongoDB collections
5. Generate embeddings for each document using Azure OpenAI
6. Insert documents with embeddings into Azure AI Search index
7. Verify search functionality with sample queries

## Vector Search Query Process

1. Generate embedding for user query using Azure OpenAI
2. Perform vector search against the embedding field
3. Apply additional filters based on document_type, categories, or fields
4. Return most relevant matches with scores
5. Post-process results to map to user requirements

## Upcoming Improvements

- Add field-level embeddings for more precise field mapping
- Create separate indexes for actions, triggers, and templates
- Implement hybrid search combining vector and keyword approaches
- Pre-compute compatibility scores between common component pairs
- Set up periodic reindexing to capture new data 
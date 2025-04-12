# MongoDB Indexes for apicus-processed-actions Collection

## Primary Index
- `_id` (default MongoDB ObjectId index)

## Secondary Indexes
- `id_` 
  - `_id_asc` (ascending)
- `action_id_1`
  - `action_id_asc` (ascending)
- `app_id_1`
  - `app_id_asc` (ascending)
- `app_name_1`
  - `app_name_asc` (ascending)
- `title_1`
  - `title_asc` (ascending)
- `type_1`
  - `type_asc` (ascending)
- `normalized_categories_1`
  - `normalized_categories_asc` (ascending)
- `embedding_vector_ivf_index`
  - `embedding_vector_cosmosSearch` (vector search index using Cosmos DB)

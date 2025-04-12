# MongoDB Indexes for apicus-processed-apps Collection

## Primary Index
- `_id` (default MongoDB ObjectId index)

## Secondary Indexes
- `id_` 
  - `_id_asc` (ascending)
- `app_id_1`
  - `app_id_asc` (ascending)
- `name_1`
  - `name_asc` (ascending)
- `slug_1`
  - `slug_asc` (ascending)
- `normalized_categories_1`
  - `normalized_categories_asc` (ascending)
- `embedding_vector_ivf_index`
  - `embedding_vector_cosmosSearch` (vector search index using Cosmos DB)

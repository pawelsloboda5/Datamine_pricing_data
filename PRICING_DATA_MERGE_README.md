# Pricing Data Merge Tool

This tool merges pricing data from the `apicus-apps-prices` collection into the corresponding documents in the `apicus-processed-apps` collection in MongoDB.

## Overview

The tool performs the following steps:

1. Creates a backup of the processed apps collection (optional)
2. Retrieves all documents from the pricing collection
3. For each pricing document, tries to find a matching document in the processed apps collection
4. Adds the pricing data to the matching document
5. Generates a CSV report of the operation

## Prerequisites

Ensure you have the following dependencies installed:
- Python 3.6+
- pymongo
- python-dotenv

These should already be in the requirements.txt file.

## Configuration

The script uses environment variables from the `.env` file:
- `MONGODB_URI` - MongoDB connection string
- `MONGODB_DB_NAME` - MongoDB database name (defaults to "apicus-db-data")

## Usage

Run the script using:

```bash
python run_merge_pricing.py [options]
```

Or make it executable and run directly:

```bash
chmod +x run_merge_pricing.py
./run_merge_pricing.py [options]
```

### Options

- `--no-backup`: Skip creating a backup of the processed apps collection
- `--dry-run`: Run in dry run mode (no actual changes to the database)
- `--verbose` or `-v`: Enable verbose logging

### Examples

Run with default settings (with backup):
```bash
python run_merge_pricing.py
```

Run without creating a backup:
```bash
python run_merge_pricing.py --no-backup
```

Run in dry run mode (no actual changes):
```bash
python run_merge_pricing.py --dry-run
```

Run with verbose logging:
```bash
python run_merge_pricing.py -v
```

## Output

The script generates a CSV report file with information about each pricing document processed:
- `app_id` - The app ID from the original metadata
- `name` - The app name from the original metadata
- `slug` - The app slug from the original metadata
- `success` - Whether the update was successful
- `match_method` - How the matching document was found (by app_id, slug, or name)
- `document_id` - The ID of the updated document
- `error` - Error message if the update failed

The report file is named `pricing_merge_report_YYYYMMDDHHMMSS.csv` (or `dry_run_pricing_merge_report_YYYYMMDDHHMMSS.csv` for dry runs).

## Backup

By default, the script creates a backup of the processed apps collection before making any changes. The backup collection is named `apicus-processed-apps-backup-YYYYMMDDHHMMSS`.

To restore from a backup, you would need to use the MongoDB tools or commands to copy the data back to the original collection.

## Matching Strategy

The script uses the following strategies to find matching documents (in order):

1. Match by app_id (most precise)
2. Match by slug (if app_id match fails)
3. Match by name (if both app_id and slug matches fail)

This ensures the highest possible match rate while maintaining accuracy. 
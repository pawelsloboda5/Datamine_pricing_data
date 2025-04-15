#!/usr/bin/env python
"""
Run script for merging pricing data from apicus-apps-prices collection
to the corresponding documents in apicus-processed-apps collection.
"""

import argparse
import logging
from merge_pricing_data import main as merge_pricing_data

def setup_logging(verbose):
    """Set up logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge pricing data into processed apps collection'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating a backup of the processed apps collection'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry run mode (no actual changes to the database)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.dry_run:
        logging.info("Running in DRY RUN mode - no changes will be made to the database")
    
    logging.info("Starting pricing data merge process...")
    merge_pricing_data(create_backup=not args.no_backup, dry_run=args.dry_run)
    logging.info("Pricing data merge process completed.") 
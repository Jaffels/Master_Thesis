#!/usr/bin/env python3
"""
Automated Gap Filling Script

Reads gap_analysis.csv and attempts to fill the missing data periods
"""

import pandas as pd
from pathlib import Path
import sys
import os
from datetime import timedelta
import time

def load_gaps(gap_file):
    """Load gaps from CSV file"""
    df = pd.read_csv(gap_file)
    # Handle mixed timezones by converting to UTC first
    df['gap_start'] = pd.to_datetime(df['gap_start'], utc=True).dt.tz_convert('Europe/Zurich')
    df['gap_end'] = pd.to_datetime(df['gap_end'], utc=True).dt.tz_convert('Europe/Zurich')
    return df

def fill_gaps_for_dataset(pipeline, dataset, gaps, max_retries=3):
    """
    Fill gaps for a specific dataset
    
    Args:
        pipeline: ENTSOEAncillaryServicesPipeline instance
        dataset: Dataset name (balancing, imbalance, etc.)
        gaps: DataFrame of gaps for this dataset
        max_retries: Maximum retry attempts per gap
    """
    print(f"\n{'='*80}")
    print(f"Filling gaps for: {dataset.upper()}")
    print(f"{'='*80}")
    print(f"Total gaps to fill: {len(gaps)}\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, gap in gaps.iterrows():
        gap_num = idx + 1
        start = gap['gap_start']
        end = gap['gap_end']
        duration = gap['duration_hours']
        
        print(f"\n[{gap_num}/{len(gaps)}] Gap: {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} ({duration:.1f}h)")
        
        # Add small buffer to ensure we capture the edges
        fetch_start = start - timedelta(hours=1)
        fetch_end = end + timedelta(hours=1)
        
        # Convert to timezone-aware timestamps
        if fetch_start.tzinfo is None:
            fetch_start = fetch_start.tz_localize('Europe/Zurich')
        if fetch_end.tzinfo is None:
            fetch_end = fetch_end.tz_localize('Europe/Zurich')
        
        # Attempt to fill the gap
        success = False
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries}...", end=' ')
                
                if dataset == 'balancing':
                    result = pipeline.fetch_balancing_prices(fetch_start, fetch_end)
                elif dataset == 'imbalance':
                    result = pipeline.fetch_imbalance_prices(fetch_start, fetch_end)
                elif dataset == 'generation':
                    result = pipeline.fetch_generation_data(fetch_start, fetch_end)
                elif dataset == 'load':
                    result = pipeline.fetch_load_data(fetch_start, fetch_end)
                elif dataset == 'crossborder':
                    result = pipeline.fetch_crossborder_flows(fetch_start, fetch_end)
                else:
                    print(f"Unknown dataset type: {dataset}")
                    break
                
                if result is not None and (isinstance(result, dict) or not result.empty):
                    print("✅ Success!")
                    success = True
                    success_count += 1
                    break
                else:
                    print("⚠️  No data returned")
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                if '503' in error_str or 'timeout' in error_str or 'unavailable' in error_str:
                    print(f"⚠️  Retryable error: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                else:
                    print(f"❌ Non-retryable error: {e}")
                    break
            
            # Rate limiting between attempts
            time.sleep(2)
        
        if not success:
            fail_count += 1
            print(f"   ❌ Failed to fill this gap")
        
        # Rate limiting between gaps
        time.sleep(3)
    
    print(f"\n{'='*80}")
    print(f"Summary for {dataset}:")
    print(f"   ✅ Successfully filled: {success_count}/{len(gaps)}")
    print(f"   ❌ Failed: {fail_count}/{len(gaps)}")
    print(f"{'='*80}")
    
    return success_count, fail_count


def main():
    print("\n" + "="*80)
    print("AUTOMATED GAP FILLING")
    print("="*80)
    
    # Check for gap_analysis.csv
    possible_locations = [
        Path.cwd() / 'gap_analysis.csv',
        Path.cwd() / 'swissgrid_ancillary_data' / 'gap_analysis.csv',
        Path.home() / 'swissgrid_ancillary_data' / 'gap_analysis.csv',
    ]
    
    gap_file = None
    for loc in possible_locations:
        if loc.exists():
            gap_file = loc
            break
    
    if not gap_file:
        gap_file_input = input("\nEnter path to gap_analysis.csv: ").strip()
        gap_file = Path(gap_file_input)
        
        if not gap_file.exists():
            print(f"❌ File not found: {gap_file}")
            sys.exit(1)
    
    print(f"\n✓ Found gap analysis: {gap_file}")
    
    # Load gaps
    gaps_df = load_gaps(gap_file)
    print(f"✓ Loaded {len(gaps_df)} gaps")
    
    # Show summary
    print("\nGap summary by dataset:")
    gap_summary = gaps_df.groupby('dataset').agg({
        'duration_hours': ['count', 'sum', 'mean']
    })
    print(gap_summary)
    
    # Get API key
    api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        print("\n❌ ENTSOE_API_KEY not found!")
        print("Please set it in .env file or environment variable")
        sys.exit(1)
    
    print(f"✓ API key loaded")
    
    # Ask for confirmation
    print("\n" + "="*80)
    print("READY TO FILL GAPS")
    print("="*80)
    print("\nThis will:")
    print(f"  • Attempt to fill {len(gaps_df)} gaps")
    print(f"  • Datasets affected: {', '.join(gaps_df['dataset'].unique())}")
    print(f"  • Estimated time: {len(gaps_df) * 10 / 60:.1f} minutes")
    print("\nNote: Some gaps may not be fillable if data was never published by ENTSOE")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("Cancelled.")
        sys.exit(0)
    
    # Initialize pipeline
    from entsoe_pipeline import ENTSOEAncillaryServicesPipeline
    
    data_dir = gap_file.parent if gap_file.parent.name == 'swissgrid_ancillary_data' else './swissgrid_ancillary_data'
    
    pipeline = ENTSOEAncillaryServicesPipeline(
        api_key=api_key,
        output_dir=str(data_dir)
    )
    
    # Fill gaps by dataset
    total_success = 0
    total_fail = 0
    
    for dataset in gaps_df['dataset'].unique():
        dataset_gaps = gaps_df[gaps_df['dataset'] == dataset]
        
        success, fail = fill_gaps_for_dataset(pipeline, dataset, dataset_gaps)
        total_success += success
        total_fail += fail
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n✅ Successfully filled: {total_success}/{len(gaps_df)} gaps")
    print(f"❌ Failed to fill: {total_fail}/{len(gaps_df)} gaps")
    
    if total_fail > 0:
        print("\nNote: Failed gaps may be due to:")
        print("  • Data never published by ENTSOE")
        print("  • API temporarily unavailable")
        print("  • Data type not available for Switzerland")
        print("\nRecommendation: Re-run find_gaps.py to see updated status")
    else:
        print("\n🎉 All gaps successfully filled!")
        print("Run find_gaps.py again to verify completeness")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

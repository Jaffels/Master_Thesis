#!/usr/bin/env python3
"""
PROPER Gap Analysis - Identifies actual missing dates in your data

This checks for:
1. Missing dates within the data range
2. Large gaps between consecutive records
3. Data quality issues
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import sys

def find_data_dir():
    """Find the data directory"""
    possible_locations = [
        Path.cwd() / 'swissgrid_ancillary_data',
        Path.cwd().parent / 'swissgrid_ancillary_data',
        Path.home() / 'swissgrid_ancillary_data',
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    user_path = input("Enter path to swissgrid_ancillary_data: ").strip()
    return Path(user_path).expanduser().resolve()


def analyze_data_continuity(data_type, data_dir):
    """
    Analyze data for gaps and missing dates
    
    Returns:
        dict with gap information
    """
    dir_path = data_dir / data_type
    
    if not dir_path.exists():
        return {
            'status': 'MISSING',
            'files': 0,
            'gaps': [],
            'coverage': 0
        }
    
    parquet_files = list(dir_path.glob('*.parquet'))
    
    if len(parquet_files) == 0:
        return {
            'status': 'EMPTY',
            'files': 0,
            'gaps': [],
            'coverage': 0
        }
    
    print(f"   Loading {len(parquet_files)} files...")
    
    # Load all data
    all_data = []
    for i, file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(file)
            all_data.append(df)
            if (i + 1) % 100 == 0:
                print(f"   Loaded {i + 1}/{len(parquet_files)} files...")
        except Exception as e:
            print(f"   ⚠️  Could not read {file.name}: {e}")
    
    if not all_data:
        return {
            'status': 'ERROR',
            'files': len(parquet_files),
            'gaps': [],
            'coverage': 0
        }
    
    # Combine all data
    combined = pd.concat(all_data, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    total_records = len(combined)
    start_date = combined.index.min()
    end_date = combined.index.max()
    
    # Calculate expected number of records (15-min intervals)
    total_duration = (end_date - start_date).total_seconds()
    expected_intervals = int(total_duration / (15 * 60)) + 1
    
    # Calculate coverage
    coverage_pct = (total_records / expected_intervals * 100) if expected_intervals > 0 else 0
    
    # Find gaps (missing periods > 1 hour)
    gaps = []
    timestamps = combined.index
    
    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        if gap > timedelta(hours=1):  # Gap larger than 1 hour
            gaps.append({
                'start': timestamps[i],
                'end': timestamps[i + 1],
                'duration': gap
            })
    
    # Determine status
    if coverage_pct >= 95:
        status = 'EXCELLENT'
    elif coverage_pct >= 80:
        status = 'GOOD'
    elif coverage_pct >= 50:
        status = 'FAIR'
    else:
        status = 'POOR'
    
    return {
        'status': status,
        'files': len(parquet_files),
        'total_records': total_records,
        'expected_records': expected_intervals,
        'coverage': coverage_pct,
        'start_date': start_date,
        'end_date': end_date,
        'gaps': gaps
    }


def main():
    print("\n" + "="*80)
    print("PROPER GAP ANALYSIS - CHECKING DATE CONTINUITY")
    print("="*80)
    
    data_dir = find_data_dir()
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"\nAnalyzing: {data_dir}\n")
    
    subdirs = ['balancing', 'imbalance', 'generation', 'load', 'crossborder', 'reserves']
    
    results = {}
    
    for subdir in subdirs:
        print(f"\n{'='*80}")
        print(f"📊 Analyzing {subdir.upper()}")
        print('='*80)
        
        result = analyze_data_continuity(subdir, data_dir)
        results[subdir] = result
        
        if result['status'] in ['MISSING', 'EMPTY', 'ERROR']:
            print(f"   Status: ❌ {result['status']}")
            continue
        
        print(f"\n   Status: {result['status']}")
        print(f"   Files: {result['files']}")
        print(f"   Records: {result['total_records']:,}")
        print(f"   Expected: {result['expected_records']:,}")
        print(f"   Coverage: {result['coverage']:.2f}%")
        print(f"   Date Range: {result['start_date'].strftime('%Y-%m-%d')} to {result['end_date'].strftime('%Y-%m-%d')}")
        
        # Show gaps
        if result['gaps']:
            print(f"\n   ⚠️  FOUND {len(result['gaps'])} GAPS (> 1 hour):")
            
            # Show first 10 gaps
            for i, gap in enumerate(result['gaps'][:10]):
                duration_days = gap['duration'].days
                duration_hours = gap['duration'].seconds // 3600
                
                print(f"      {i+1}. {gap['start'].strftime('%Y-%m-%d %H:%M')} → "
                      f"{gap['end'].strftime('%Y-%m-%d %H:%M')} "
                      f"({duration_days}d {duration_hours}h)")
            
            if len(result['gaps']) > 10:
                print(f"      ... and {len(result['gaps']) - 10} more gaps")
        else:
            print(f"\n   ✅ NO GAPS - Continuous data!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Dataset':<15} {'Status':<12} {'Coverage':<12} {'Gaps':<10}")
    print("-" * 60)
    
    for subdir, result in results.items():
        if result['status'] in ['MISSING', 'EMPTY', 'ERROR']:
            print(f"{subdir:<15} {'❌ ' + result['status']:<12} {'N/A':<12} {'N/A':<10}")
        else:
            status_icon = {
                'EXCELLENT': '✅',
                'GOOD': '✓',
                'FAIR': '⚠️',
                'POOR': '❌'
            }.get(result['status'], '?')
            
            print(f"{subdir:<15} {status_icon + ' ' + result['status']:<12} "
                  f"{result['coverage']:.1f}%{' ':<7} {len(result['gaps']):<10}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    excellent = [k for k, v in results.items() if v['status'] == 'EXCELLENT']
    good = [k for k, v in results.items() if v['status'] == 'GOOD']
    fair = [k for k, v in results.items() if v['status'] == 'FAIR']
    poor = [k for k, v in results.items() if v['status'] in ['POOR', 'MISSING', 'EMPTY']]
    
    if len(excellent) + len(good) >= 3:
        print("\n✅ YOU HAVE SUFFICIENT DATA FOR YOUR THESIS!")
        print(f"\n   Usable datasets ({len(excellent) + len(good)}):")
        for dataset in excellent + good:
            cov = results[dataset].get('coverage', 0)
            print(f"      • {dataset}: {cov:.1f}% coverage")
        
        print("\n   Next steps:")
        print("      1. Run: python3 data_processing.py")
        print("      2. Use available datasets for forecasting")
        print("      3. Mention data gaps in thesis methodology section")
        
        if fair:
            print(f"\n   ⚠️  Fair quality datasets ({len(fair)}): {', '.join(fair)}")
            print("      Consider using with caution or excluding from analysis")
    
    else:
        print("\n⚠️  LIMITED DATA AVAILABLE")
        print("\n   Options:")
        print("      1. Focus analysis on available datasets only")
        print("      2. Retry data collection during off-peak hours (2-6 AM CET)")
        print("      3. Use shorter time period (e.g., last 2 years only)")
    
    if poor:
        print(f"\n   ❌ Insufficient/missing datasets: {', '.join(poor)}")
        print("      Note: 'reserves' data often not published via ENTSOE API")
    
    # Export gaps to file
    print("\n" + "="*80)
    
    export = input("\nExport gap details to CSV? (y/n): ").strip().lower()
    if export in ['y', 'yes']:
        output_file = data_dir / 'gap_analysis.csv'
        
        gap_data = []
        for dataset, result in results.items():
            if result.get('gaps'):
                for gap in result['gaps']:
                    gap_data.append({
                        'dataset': dataset,
                        'gap_start': gap['start'],
                        'gap_end': gap['end'],
                        'duration_hours': gap['duration'].total_seconds() / 3600
                    })
        
        if gap_data:
            gap_df = pd.DataFrame(gap_data)
            gap_df.to_csv(output_file, index=False)
            print(f"✅ Exported to: {output_file}")
        else:
            print("No gaps to export!")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

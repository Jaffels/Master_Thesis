#!/usr/bin/env python3
"""
Frequency-Aware Gap Analysis

Detects actual data frequency (15-min, hourly, daily) and calculates
coverage correctly for each dataset
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import sys
import numpy as np

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


def detect_frequency(timestamps):
    """
    Detect the actual data frequency from timestamps
    
    Returns:
        str: 'hourly', '15min', '30min', 'daily', etc.
    """
    if len(timestamps) < 10:
        return 'unknown'
    
    # Calculate differences between consecutive timestamps
    diffs = pd.Series(timestamps).diff().dropna()
    
    # Get the most common interval (mode)
    most_common = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs.median()
    
    # Convert to minutes
    minutes = most_common.total_seconds() / 60
    
    if 14 <= minutes <= 16:
        return '15min'
    elif 29 <= minutes <= 31:
        return '30min'
    elif 59 <= minutes <= 61:
        return 'hourly'
    elif 1430 <= minutes <= 1450:
        return 'daily'
    else:
        return f'{int(minutes)}min'


def analyze_with_frequency(data_type, data_dir):
    """
    Analyze data detecting its actual frequency
    """
    dir_path = data_dir / data_type
    
    if not dir_path.exists():
        return {
            'status': 'MISSING',
            'files': 0,
            'frequency': 'N/A',
            'gaps': [],
            'coverage': 0
        }
    
    parquet_files = list(dir_path.glob('*.parquet'))
    
    if len(parquet_files) == 0:
        return {
            'status': 'EMPTY',
            'files': 0,
            'frequency': 'N/A',
            'gaps': [],
            'coverage': 0
        }
    
    print(f"   Loading {len(parquet_files)} files...")
    
    # Load sample to detect frequency
    sample_files = parquet_files[:min(5, len(parquet_files))]
    sample_data = []
    
    for file in sample_files:
        try:
            df = pd.read_parquet(file)
            sample_data.append(df)
        except:
            continue
    
    if not sample_data:
        return {
            'status': 'ERROR',
            'files': len(parquet_files),
            'frequency': 'N/A',
            'gaps': [],
            'coverage': 0
        }
    
    sample_combined = pd.concat(sample_data, axis=0).sort_index()
    frequency = detect_frequency(sample_combined.index)
    
    print(f"   Detected frequency: {frequency}")
    
    # Now load all data
    all_data = []
    for i, file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(file)
            all_data.append(df)
            if (i + 1) % 100 == 0:
                print(f"   Loaded {i + 1}/{len(parquet_files)} files...")
        except Exception as e:
            print(f"   ⚠️  Could not read {file.name}")
    
    combined = pd.concat(all_data, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    total_records = len(combined)
    start_date = combined.index.min()
    end_date = combined.index.max()
    
    # Calculate expected records based on detected frequency
    total_duration = (end_date - start_date).total_seconds()
    
    if frequency == '15min':
        expected_intervals = int(total_duration / (15 * 60)) + 1
    elif frequency == '30min':
        expected_intervals = int(total_duration / (30 * 60)) + 1
    elif frequency == 'hourly':
        expected_intervals = int(total_duration / (60 * 60)) + 1
    elif frequency == 'daily':
        expected_intervals = int(total_duration / (24 * 60 * 60)) + 1
    else:
        # Default to 15min
        expected_intervals = int(total_duration / (15 * 60)) + 1
    
    # Calculate coverage
    coverage_pct = (total_records / expected_intervals * 100) if expected_intervals > 0 else 0
    
    # Find gaps (threshold depends on frequency)
    if frequency == '15min':
        gap_threshold = timedelta(hours=1)
    elif frequency == 'hourly':
        gap_threshold = timedelta(hours=3)
    elif frequency == 'daily':
        gap_threshold = timedelta(days=2)
    else:
        gap_threshold = timedelta(hours=1)
    
    gaps = []
    timestamps = combined.index
    
    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]
        if gap > gap_threshold:
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
        'frequency': frequency,
        'start_date': start_date,
        'end_date': end_date,
        'gaps': gaps
    }


def main():
    print("\n" + "="*80)
    print("FREQUENCY-AWARE GAP ANALYSIS")
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
        
        result = analyze_with_frequency(subdir, data_dir)
        results[subdir] = result
        
        if result['status'] in ['MISSING', 'EMPTY', 'ERROR']:
            print(f"   Status: ❌ {result['status']}")
            continue
        
        print(f"\n   Status: {result['status']}")
        print(f"   Frequency: {result['frequency']}")
        print(f"   Files: {result['files']}")
        print(f"   Records: {result['total_records']:,}")
        print(f"   Expected: {result['expected_records']:,}")
        print(f"   Coverage: {result['coverage']:.2f}%")
        print(f"   Date Range: {result['start_date'].strftime('%Y-%m-%d')} to {result['end_date'].strftime('%Y-%m-%d')}")
        
        if result['gaps']:
            print(f"\n   ⚠️  FOUND {len(result['gaps'])} GAPS:")
            for i, gap in enumerate(result['gaps'][:5]):
                duration_days = gap['duration'].days
                duration_hours = gap['duration'].seconds // 3600
                print(f"      {i+1}. {gap['start'].strftime('%Y-%m-%d %H:%M')} → "
                      f"{gap['end'].strftime('%Y-%m-%d %H:%M')} "
                      f"({duration_days}d {duration_hours}h)")
            if len(result['gaps']) > 5:
                print(f"      ... and {len(result['gaps']) - 5} more")
        else:
            print(f"\n   ✅ NO GAPS - Continuous data!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Dataset':<15} {'Frequency':<10} {'Coverage':<12} {'Status':<12} {'Gaps':<10}")
    print("-" * 70)
    
    for subdir, result in results.items():
        if result['status'] in ['MISSING', 'EMPTY', 'ERROR']:
            print(f"{subdir:<15} {'N/A':<10} {'N/A':<12} {'❌ ' + result['status']:<12} {'N/A':<10}")
        else:
            status_icon = {
                'EXCELLENT': '✅',
                'GOOD': '✓',
                'FAIR': '⚠️',
                'POOR': '❌'
            }.get(result['status'], '?')
            
            print(f"{subdir:<15} {result['frequency']:<10} "
                  f"{result['coverage']:.1f}%{' ':<7} "
                  f"{status_icon + ' ' + result['status']:<12} {len(result['gaps']):<10}")
    
    # Recommendations
    print("\n" + "="*80)
    print("ACTUAL DATA QUALITY")
    print("="*80)
    
    usable = [k for k, v in results.items() 
              if v['status'] in ['EXCELLENT', 'GOOD']]
    
    print(f"\n✅ USABLE DATASETS: {len(usable)}")
    for dataset in usable:
        freq = results[dataset]['frequency']
        cov = results[dataset]['coverage']
        print(f"   • {dataset} ({freq}): {cov:.1f}%")
    
    if len(usable) >= 3:
        print(f"\n🎉 YOU HAVE EXCELLENT DATA!")
        print(f"   Ready for: python3 data_processing.py")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Quick Start Script for ENTSOE Ancillary Services Data Pipeline

This script provides an interactive way to:
1. Test API connection
2. Run data collection
3. Process and analyze data
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

def setup_api_key():
    """Guide user through API key setup"""
    print("=" * 60)
    print("ENTSOE API Key Setup")
    print("=" * 60)
    
    if not Path('.env').exists():
        print("\n.env file not found.")
        print("\n1. Get your API key from: https://transparency.entsoe.eu/")
        print("2. Register for an account if you haven't already")
        print("3. Navigate to 'My Account' > 'Web API Security Token'")
        
        api_key = input("\nEnter your ENTSOE API key: ").strip()
        
        with open('.env', 'w') as f:
            f.write(f'ENTSOE_API_KEY={api_key}\n')
        
        print("\n✓ API key saved to .env file")
    else:
        print("\n✓ .env file already exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ENTSOE_API_KEY')
    if not api_key:
        print("\n✗ Error: ENTSOE_API_KEY not found in .env file")
        return None
    
    return api_key


def test_api_connection(api_key):
    """Test API connection with a small query"""
    print("\n" + "=" * 60)
    print("Testing API Connection")
    print("=" * 60)
    
    try:
        from entsoe import EntsoePandasClient
        import pandas as pd
        
        client = EntsoePandasClient(api_key=api_key)
        
        # Test with recent 1-day query for Swiss load
        end = pd.Timestamp.now(tz='Europe/Zurich')
        start = end - timedelta(days=1)
        
        print(f"\nTesting with load data for {start.date()} to {end.date()}...")
        
        data = client.query_load('CH', start=start, end=end)
        
        if data is not None and not data.empty:
            print(f"\n✓ Connection successful!")
            print(f"  Retrieved {len(data)} data points")
            print(f"  Date range: {data.index.min()} to {data.index.max()}")
            return True
        else:
            print("\n✗ Connection successful but no data returned")
            print("  This may be normal if data is not available for recent dates")
            return True
            
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - Network connectivity")
        print("  - ENTSOE API temporarily unavailable")
        return False


def run_data_collection(api_key):
    """Run the full data collection pipeline"""
    print("\n" + "=" * 60)
    print("Data Collection Configuration")
    print("=" * 60)
    
    # Get date range
    print("\nRecommended data range for thesis: 2-3 years")
    
    default_years = 2
    years_str = input(f"\nHow many years of data to collect? (default: {default_years}): ").strip()
    
    if years_str:
        try:
            years = int(years_str)
        except ValueError:
            print("Invalid input, using default")
            years = default_years
    else:
        years = default_years
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    proceed = input("\nProceed with data collection? (y/n): ").strip().lower()
    
    if proceed != 'y':
        print("Collection cancelled")
        return
    
    # Run pipeline
    print("\n" + "=" * 60)
    print("Starting Data Collection")
    print("=" * 60)
    print("\nThis may take 30-60 minutes depending on date range...")
    print("Progress will be logged below:\n")
    
    from entsoe_pipeline import ENTSOEAncillaryServicesPipeline
    
    pipeline = ENTSOEAncillaryServicesPipeline(
        api_key=api_key,
        output_dir='./swissgrid_ancillary_data'
    )
    
    pipeline.run_full_collection(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        chunk_days=7
    )
    
    print("\n" + "=" * 60)
    print("Data Collection Complete!")
    print("=" * 60)


def process_data():
    """Process collected data and create features"""
    print("\n" + "=" * 60)
    print("Data Processing & Feature Engineering")
    print("=" * 60)
    
    from data_processing import AncillaryServicesFeatureEngineer
    
    engineer = AncillaryServicesFeatureEngineer(
        data_dir='./swissgrid_ancillary_data'
    )
    
    # Create master dataset
    print("\nCreating master dataset...")
    master = engineer.create_master_dataset()
    
    if master.empty:
        print("\n✗ No data available to process")
        print("  Please run data collection first (option 2)")
        return
    
    print(f"✓ Master dataset created: {master.shape}")
    
    # Identify available price columns
    price_cols = [col for col in master.columns if 'price' in col.lower()]
    
    if not price_cols:
        print("\n✗ No price columns found in data")
        return
    
    print(f"\nAvailable price columns: {price_cols[:5]}")
    
    # Prepare forecasting dataset
    target_col = price_cols[0]
    print(f"\nPreparing forecasting dataset for: {target_col}")
    
    X, y = engineer.prepare_forecasting_dataset(
        target_col=target_col,
        forecast_horizon=4  # 1 hour ahead
    )
    
    print(f"✓ Forecasting dataset prepared")
    print(f"  Features: {X.shape}")
    print(f"  Target: {y.shape}")
    
    # Response time dataset
    print("\nCreating response time dataset...")
    response_df = engineer.create_response_time_dataset()
    
    if not response_df.empty:
        print(f"✓ Response time dataset created: {response_df.shape}")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nProcessed data saved to: ./swissgrid_ancillary_data/processed/")


def main():
    """Main interactive menu"""
    print("\n" + "=" * 60)
    print("ENTSOE Ancillary Services Data Pipeline")
    print("Master's Thesis Quick Start")
    print("=" * 60)
    
    while True:
        print("\n\nWhat would you like to do?\n")
        print("1. Setup API key")
        print("2. Test API connection")
        print("3. Run data collection")
        print("4. Process collected data")
        print("5. Launch Jupyter notebook for analysis")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            setup_api_key()
        
        elif choice == '2':
            api_key = os.getenv('ENTSOE_API_KEY')
            if not api_key:
                print("\n✗ API key not set. Please run option 1 first.")
            else:
                test_api_connection(api_key)
        
        elif choice == '3':
            api_key = os.getenv('ENTSOE_API_KEY')
            if not api_key:
                print("\n✗ API key not set. Please run option 1 first.")
            else:
                run_data_collection(api_key)
        
        elif choice == '4':
            if not Path('./swissgrid_ancillary_data').exists():
                print("\n✗ No data directory found. Please run collection first (option 3).")
            else:
                process_data()
        
        elif choice == '5':
            print("\nLaunching Jupyter notebook...")
            import subprocess
            subprocess.run(['jupyter', 'notebook', 'exploratory_analysis.ipynb'])
        
        elif choice == '6':
            print("\nGoodbye! Good luck with your thesis! 🚀")
            break
        
        else:
            print("\nInvalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()

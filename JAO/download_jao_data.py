#!/usr/bin/env python3
"""
JAO Data Downloader for Switzerland
Downloads all available data from JAO (Joint Allocation Office).

IMPORTANT - API KEY REQUIRED:
The JAO Publication Tool API requires an API key (AUTH_API_KEY).
To request one:
  1. Go to: https://www.jao.eu/
  2. Open a support ticket under 'Technical Support'
  3. Request an AUTH_API_KEY for the Publication Tool API

DATA AVAILABILITY:
  - Core Day-Ahead data:     2022-06-09 onwards
  - Core Intraday (b):       2024-05-29 onwards
  - Core Intraday (a):       2024-06-14 onwards
  - Auction/NTC history:     2006 onwards (via JaoAPIClient, separate key)

SWISS BORDERS: CH-FR, CH-IT, CH-AT, CH-DE

Rate Limit: 100 requests/minute (JAO enforced)
"""

import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JAODataDownloader:

    def __init__(self, api_key: str, output_dir: str = './jao_data'):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.swiss_borders = [
            ('CH', 'FR'),
            ('CH', 'IT'),
            ('CH', 'AT'),
            ('CH', 'DE'),
        ]

        # Rate limiting: 100 requests/minute → 0.7s between requests (conservative)
        self.min_delay = 0.7
        self.last_request_time = 0
        self.request_count = 0

        self.download_report = {
            'timestamp': datetime.now().isoformat(),
            'downloads': {}
        }

        self.subdirs = {
            'scheduled_exchanges':  self.output_dir / 'scheduled_exchanges',
            'congestion_income':    self.output_dir / 'congestion_income',
            'price_spread':         self.output_dir / 'price_spread',
            'lta':                  self.output_dir / 'lta',
            'minmax_np':            self.output_dir / 'minmax_net_positions',
            'maxbex':               self.output_dir / 'max_bilateral_exchanges',
            'net_position':         self.output_dir / 'net_positions',
            'alloc_constraint':     self.output_dir / 'allocation_constraints',
            'refprog':              self.output_dir / 'reference_programme',
            'active_constraints':   self.output_dir / 'active_constraints',
            'ntc_intraday':         self.output_dir / 'ntc_intraday',
            'atc_intraday':         self.output_dir / 'atc_intraday',
            'auctions':             self.output_dir / 'auctions',
        }
        for d in self.subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
        if self.request_count % 20 == 0:
            logger.info(f"  [{self.request_count} requests made]")

    def save(self, df: pd.DataFrame, subdir_key: str, name: str) -> bool:
        if df is None or df.empty:
            logger.warning(f"  No data returned for {name}")
            return False
        path = self.subdirs[subdir_key] / f"{name}.parquet"
        df.to_parquet(path)
        logger.info(f"  ✓ Saved {name} ({len(df)} rows) → {path.name}")
        return True

    def record(self, category: str, key: str, success: bool, msg: str, rows: int = 0):
        self.download_report['downloads'].setdefault(category, {})[key] = {
            'success': success, 'message': msg,
            'records': rows, 'timestamp': datetime.now().isoformat()
        }

    def run(self, fn, category: str, key: str, subdir_key: str, name: str):
        """Execute a query function, save result, record outcome."""
        try:
            self.rate_limit()
            df = fn()
            saved = self.save(df, subdir_key, name)
            rows = len(df) if df is not None else 0
            self.record(category, key, saved, 'OK' if saved else 'Empty response', rows)
        except Exception as e:
            msg = str(e)[:200]
            logger.error(f"  ✗ {key}: {msg}")
            self.record(category, key, False, msg)

    def date_chunks(self, start: pd.Timestamp, end: pd.Timestamp, days: int = 30):
        """Split a date range into chunks to avoid timeouts on large queries."""
        current = start
        while current < end:
            chunk_end = min(current + pd.Timedelta(days=days), end)
            yield current, chunk_end
            current = chunk_end + pd.Timedelta(days=1)

    # ------------------------------------------------------------------
    # Core Day-Ahead client queries (from=2022-06-09)
    # ------------------------------------------------------------------

    def download_da_data(self, start: pd.Timestamp, end: pd.Timestamp):
        logger.info("\n" + "="*70)
        logger.info("CORE DAY-AHEAD DATA  (available from 2022-06-09)")
        logger.info("="*70)

        from jao import JaoPublicationToolPandasClient
        client = JaoPublicationToolPandasClient(api_key=self.api_key)

        # Scheduled Exchanges (all borders in one call - no border filter)
        logger.info("\n[Scheduled Exchanges]")
        for s, e in self.date_chunks(start, end):
            tag = f"scheduled_exchanges_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_scheduled_exchange(s, e),
                'scheduled_exchanges', tag, 'scheduled_exchanges', tag
            )

        # Congestion Income
        logger.info("\n[Congestion Income]")
        for s, e in self.date_chunks(start, end):
            tag = f"congestion_income_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_congestion_income(s, e),
                'congestion_income', tag, 'congestion_income', tag
            )

        # Price Spread
        logger.info("\n[Price Spread]")
        for s, e in self.date_chunks(start, end):
            tag = f"price_spread_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_price_spread(s, e),
                'price_spread', tag, 'price_spread', tag
            )

        # Long Term Allocation
        logger.info("\n[Long Term Allocation (LTA)]")
        for s, e in self.date_chunks(start, end):
            tag = f"lta_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_lta(s, e),
                'lta', tag, 'lta', tag
            )

        # Net Positions (date range)
        logger.info("\n[Net Positions]")
        for s, e in self.date_chunks(start, end):
            tag = f"net_position_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_net_position_fromto(s, e),
                'net_position', tag, 'net_position', tag
            )

        # Allocation Constraints
        logger.info("\n[Allocation Constraints]")
        for s, e in self.date_chunks(start, end):
            tag = f"alloc_constraint_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_allocationconstraint(s, e),
                'alloc_constraint', tag, 'alloc_constraint', tag
            )

        # Reference Programme
        logger.info("\n[Reference Programme]")
        for s, e in self.date_chunks(start, end):
            tag = f"refprog_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
            self.run(
                lambda s=s, e=e: client.query_refprog(s, e),
                'refprog', tag, 'refprog', tag
            )

        # Per-day queries: Min/Max Net Positions, Max Bilateral Exchanges, Active Constraints
        logger.info("\n[Per-Day: MinMax Net Positions, Max Bilateral Exchanges, Active Constraints]")
        current = start
        while current <= end:
            day_str = current.strftime('%Y%m%d')

            self.run(
                lambda d=current: client.query_minmax_np(d),
                'minmax_np', f'minmax_np_{day_str}', 'minmax_np', f'minmax_np_{day_str}'
            )
            self.run(
                lambda d=current: client.query_maxbex(d),
                'maxbex', f'maxbex_{day_str}', 'maxbex', f'maxbex_{day_str}'
            )
            self.run(
                lambda d=current: client.query_active_constraints(d),
                'active_constraints', f'active_constraints_{day_str}',
                'active_constraints', f'active_constraints_{day_str}'
            )
            current += pd.Timedelta(days=1)

        # Per-day per-border: Swiss NTC intraday (sidc_ntc on DA client)
        logger.info("\n[Per-Day Per-Border: NTC Swiss borders]")
        current = start
        while current <= end:
            day_str = current.strftime('%Y%m%d')
            for from_z, to_z in self.swiss_borders:
                name = f"ntc_{from_z}_{to_z}_{day_str}"
                self.run(
                    lambda d=current, f=from_z, t=to_z: client.query_minmax_np(d),
                    'ntc_intraday', name, 'ntc_intraday', name
                )
            current += pd.Timedelta(days=1)

    # ------------------------------------------------------------------
    # Core Intraday client queries
    # ------------------------------------------------------------------

    def download_intraday_data(self, start: pd.Timestamp, end: pd.Timestamp):
        logger.info("\n" + "="*70)
        logger.info("CORE INTRADAY DATA")
        logger.info("  IDCC(b): available from 2024-05-29")
        logger.info("  IDCC(a): available from 2024-06-14")
        logger.info("="*70)

        from jao import JaoPublicationToolPandasIntraDay

        for version in ['b', 'a']:
            logger.info(f"\n[IDCC({version})]")
            go_live = {
                'b': pd.Timestamp('2024-05-29', tz='UTC'),
                'a': pd.Timestamp('2024-06-14', tz='UTC'),
            }[version]

            effective_start = max(start, go_live)
            if effective_start > end:
                logger.warning(f"  Skipping IDCC({version}): requested range is before go-live {go_live.date()}")
                continue

            client = JaoPublicationToolPandasIntraDay(version=version, api_key=self.api_key)
            current = effective_start

            while current <= end:
                day_str = current.strftime('%Y%m%d')

                # SIDC NTC per border
                for from_z, to_z in self.swiss_borders:
                    name = f"sidc_ntc_{version}_{from_z}_{to_z}_{day_str}"
                    self.run(
                        lambda d=current, f=from_z, t=to_z: client.query_sidc_ntc(d, from_zone=f, to_zone=t),
                        'ntc_intraday', name, 'ntc_intraday', name
                    )
                    name = f"sidc_atc_{version}_{from_z}_{to_z}_{day_str}"
                    self.run(
                        lambda d=current, f=from_z, t=to_z: client.query_sidc_atc(d, from_zone=f, to_zone=t),
                        'atc_intraday', name, 'atc_intraday', name
                    )

                current += pd.Timedelta(days=1)

    # ------------------------------------------------------------------
    # Auction API client (long history from 2006)
    # ------------------------------------------------------------------

    def download_auction_data(self, start: pd.Timestamp, end: pd.Timestamp):
        logger.info("\n" + "="*70)
        logger.info("AUCTION DATA  (available from ~2006, requires separate API key)")
        logger.info("="*70)

        from jao import JaoAPIClient
        try:
            client = JaoAPIClient(api_key=self.api_key)
        except Exception as e:
            logger.error(f"  Could not initialize JaoAPIClient: {e}")
            logger.warning("  The Auction API may require a separate API key from JAO support.")
            self.record('auctions', 'init', False, str(e))
            return

        # Get available corridors
        logger.info("\n[Auction Corridors]")
        try:
            self.rate_limit()
            corridors = client.query_auction_corridors()
            logger.info(f"  Available corridors: {corridors}")

            # Filter for Swiss corridors
            swiss_corridor_names = ['CH-FR', 'CH-IT', 'CH-AT', 'CH-DE',
                                    'FR-CH', 'IT-CH', 'AT-CH', 'DE-CH']
            swiss_corridors = [c for c in corridors if any(s in str(c) for s in swiss_corridor_names)]
            logger.info(f"  Swiss corridors found: {swiss_corridors}")
        except Exception as e:
            logger.error(f"  Could not fetch corridors: {e}")
            self.record('auctions', 'corridors', False, str(e))
            return

        # Auction horizons
        logger.info("\n[Auction Horizons]")
        try:
            self.rate_limit()
            horizons = client.query_auction_horizons()
            logger.info(f"  Available horizons: {horizons}")
        except Exception as e:
            logger.error(f"  Could not fetch horizons: {e}")
            horizons = ['Monthly', 'Yearly', 'Daily']

        # Download auction stats per Swiss corridor and month
        logger.info("\n[Auction Stats - Monthly aggregates per Swiss corridor]")
        current = start.to_period('M').to_timestamp()
        while current <= end:
            month_date = current.date()
            for corridor in swiss_corridors:
                for horizon in ['Monthly', 'Yearly']:
                    name = f"auction_stats_{corridor}_{horizon}_{current.strftime('%Y%m')}"
                    try:
                        self.rate_limit()
                        df = client.query_auction_stats_months(
                            month_from=month_date,
                            month_to=month_date,
                            corridor=str(corridor),
                            horizon=horizon
                        )
                        saved = self.save(df, 'auctions', name)
                        self.record('auctions', name, saved, 'OK' if saved else 'Empty', len(df) if df is not None else 0)
                    except Exception as e:
                        msg = str(e)[:200]
                        logger.error(f"  ✗ {name}: {msg}")
                        self.record('auctions', name, False, msg)

            current += pd.DateOffset(months=1)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def download_all(self, start_date: str, end_date: str):
        start = pd.Timestamp(start_date, tz='UTC')
        end = pd.Timestamp(end_date, tz='UTC')

        # Enforce minimum start date for DA data
        da_go_live = pd.Timestamp('2022-06-09', tz='UTC')

        logger.info("\n" + "="*70)
        logger.info("JAO DATA DOWNLOADER FOR SWITZERLAND")
        logger.info("="*70)
        logger.info(f"Requested period : {start_date} → {end_date}")
        logger.info(f"Borders          : {[f'{f}-{t}' for f,t in self.swiss_borders]}")
        logger.info(f"Output           : {self.output_dir}")
        logger.info(f"Rate limit       : 100 req/min (0.7s enforced delay)")
        logger.info("="*70)

        # Day-Ahead data (2022-06-09 onwards)
        if end >= da_go_live:
            da_start = max(start, da_go_live)
            if da_start > start:
                logger.warning(f"Day-ahead data only available from {da_go_live.date()}, adjusting start.")
            self.download_da_data(da_start, end)
        else:
            logger.warning(f"Requested range is before Core DA go-live ({da_go_live.date()}). Skipping DA data.")

        # Intraday data (2024-05-29 / 2024-06-14 onwards)
        self.download_intraday_data(start, end)

        # Auction data (2006 onwards, separate API key may be needed)
        self.download_auction_data(start, end)

        self.generate_report()

    def generate_report(self):
        report_path = self.output_dir / 'jao_download_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.download_report, f, indent=2)

        total_ok = sum(
            1 for cat in self.download_report['downloads'].values()
            for item in cat.values() if item['success']
        )
        total = sum(len(cat) for cat in self.download_report['downloads'].values())
        total_rows = sum(
            item.get('records', 0)
            for cat in self.download_report['downloads'].values()
            for item in cat.values()
        )

        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"  Successful : {total_ok}/{total}")
        logger.info(f"  Total rows : {total_rows:,}")
        logger.info(f"  API calls  : {self.request_count}")
        logger.info(f"  Report     : {report_path}")
        logger.info("="*70)


# ------------------------------------------------------------------
# Dependency check
# ------------------------------------------------------------------

def check_dependencies():
    logger.info("Checking dependencies...")
    logger.info(f"Python: {sys.executable}  ({sys.version.split()[0]})")
    try:
        import jao
        logger.info("✓ jao-py is installed")
        return True
    except ImportError:
        pass
    result = subprocess.run(
        [sys.executable, "-c", "import jao; print(jao.__file__)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.warning(f"⚠ jao-py found but not importable. Try: {sys.executable} -m pip install --force-reinstall jao-py")
        return True
    logger.error("✗ jao-py NOT installed")
    logger.error(f"  Fix: {sys.executable} -m pip install jao-py")
    return False


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("JAO DATA DOWNLOADER FOR SWITZERLAND")
    print("Borders: CH-FR, CH-IT, CH-AT, CH-DE")
    print("="*70)

    if not check_dependencies():
        return

    print("""
IMPORTANT: A JAO API key (AUTH_API_KEY) is required.
To request one:
  1. Go to https://www.jao.eu/
  2. Open a support ticket under 'Technical Support'
  3. Request an AUTH_API_KEY for the Publication Tool API

You can also set it as an environment variable: JAO_API_KEY
""")

    api_key = os.environ.get('JAO_API_KEY', '').strip()
    if not api_key:
        api_key = input("Enter your JAO API key (or press Enter to skip auth): ").strip()

    start_date = input("Start date (YYYY-MM-DD, default: 2024-01-01): ").strip() or '2024-01-01'
    end_date   = input("End date   (YYYY-MM-DD, default: 2024-12-31): ").strip() or '2024-12-31'
    output_dir = input("Output dir (default: ./jao_data): ").strip() or './jao_data'

    print(f"\nPeriod : {start_date} → {end_date}")
    print(f"Output : {output_dir}")
    print(f"API key: {'set' if api_key else 'NOT SET (will get 403 errors)'}")

    if input("\nProceed? (yes/no): ").strip().lower() not in ('yes', 'y'):
        print("Cancelled.")
        return

    downloader = JAODataDownloader(api_key=api_key, output_dir=output_dir)
    downloader.download_all(start_date, end_date)


if __name__ == "__main__":
    main()

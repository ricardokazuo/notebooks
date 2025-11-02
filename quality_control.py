# =============================================================================
# S&P 500 STOCK DATA VALIDATION SCRIPT
# =============================================================================
# This script validates historical stock price data stored in a PostgreSQL 
# database by comparing it against fresh data downloaded from Yahoo Finance.
#
# Main Purpose:
# - Verify the accuracy of stock data in the database
# - Download reference data from yfinance for comparison
# - Identify and report any mismatches between database and yfinance data
#
# Workflow Control Flags (at top of script):
# - Set these to True/False to control which operations run
# =============================================================================


import psycopg2
from psycopg2 import sql
from datetime import datetime
import warnings
import yfinance as yf
import os
import pandas as pd

warnings.filterwarnings('ignore')
DROP_TABLE = False
CREATE_TABLE = False
DISPLAY_DATA = False
RUN_QUERY = False
DOWNLOAD_YFINANCE = False
VALIDATE_DATA = True

def load_config():
    """Load database configuration from db_config.py"""
    try:
        # Check if already loaded
        global DB_CONFIG, DATA_PATH, TABLE_NAME, TABLE_INFO, TABLE_SP500
        DB_CONFIG
        print("✅ Configuration already loaded")
    except NameError:
        print("⚠️ Loading configuration from db_config.py...")
        from db_config import (  # type: ignore
            DB_CONFIG as _DB_CONFIG,
            DATA_PATH as _DATA_PATH,
            TABLE_NAME as _TABLE_NAME,
            TABLE_INFO as _TABLE_INFO,
            TABLE_SP500 as _TABLE_SP500
        )
        # Make them global so they're accessible everywhere
        DB_CONFIG = _DB_CONFIG
        DATA_PATH = _DATA_PATH
        TABLE_NAME = _TABLE_NAME
        TABLE_INFO = _TABLE_INFO
        TABLE_SP500 = _TABLE_SP500
        
        print("✅ Configuration loaded successfully!")
        print(f"   Database: {DB_CONFIG['dbname']}")
        print(f"   Host: {DB_CONFIG['host']}")

# =====================================================================
# DATABASE FUNCTIONS
# =====================================================================

def display_summary(conn, table_name):
    """Display summary statistics"""
    try:
        with conn.cursor() as cur:
            # Top 10 by weight
            cur.execute(f"""
                SELECT ticker, weight 
                FROM {table_name} 
                ORDER BY weight DESC 
                LIMIT 10;
            """)
            top_10 = cur.fetchall()
            
            # Statistics
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_tickers,
                    SUM(weight) as total_weight,
                    AVG(weight) as avg_weight,
                    MAX(weight) as max_weight,
                    MIN(weight) as min_weight
                FROM {table_name};
            """)
            stats = cur.fetchone()
            
        print("\n" + "=" * 60)
        print("TOP 10 S&P 500 COMPANIES BY WEIGHT")
        print("=" * 60)
        for ticker, weight in top_10:
            print(f"  {ticker:8} {weight:6.2f}%")
        
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"  Total Tickers:  {stats[0]}")
        print(f"  Total Weight:   {stats[1]:.2f}%")
        print(f"  Average Weight: {stats[2]:.4f}%")
        print(f"  Highest Weight: {stats[3]:.2f}%")
        print(f"  Lowest Weight:  {stats[4]:.2f}%")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to display summary: {e}")

def run_query(conn, table_name):
    """Quality control query - Last close value per ticker per year"""
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                WITH stock_data AS (
                    SELECT
                        Ticker,
                        Date,
                        Close,
                        EXTRACT(YEAR FROM Date) AS year
                    FROM {table_name}
                    WHERE Ticker IN (SELECT ticker FROM {table_name})
                      AND Close IS NOT NULL
                ),
                ranked_data AS (
                    SELECT
                        Ticker,
                        Date,
                        Close,
                        year,
                        ROW_NUMBER() OVER (
                            PARTITION BY Ticker, year 
                            ORDER BY Date DESC
                        ) AS rn
                    FROM stock_data
                )
                SELECT 
                    Ticker,
                    year,
                    Date AS last_date,
                    Close AS last_close
                FROM ranked_data
                WHERE rn = 1
                ORDER BY Ticker, year;
            """)
            
            results = cur.fetchall()
            
        print("\n" + "=" * 80)
        print("QUALITY CONTROL: Last Close Value Per Ticker Per Year")
        print("=" * 80)
        print(f"{'Ticker':<10} {'Year':<8} {'Last Date':<15} {'Last Close':<12}")
        print("-" * 80)
        for row in results:
            print(f"{row[0]:<10} {int(row[1]):<8} {row[2]} ${row[3]:>10.2f}")
        print("=" * 80)
        print(f"Total records: {len(results)}")
        
    except Exception as e:
        print(f"❌ Failed to run query: {e}")

def download_yfinance_data(conn, table_name , specific_ticker='AAPL'):
    """
    Download historical data from yfinance for S&P 500 tickers
    
    Parameters:
    -----------
    conn : psycopg2.connection
        Database connection
    specific_ticker : str or None
        Ticker to download. Default is 'AAPL'. 
        Set to None to download all S&P 500 tickers.
    """
    try:
        # Get tickers from sp500_tickers table
        with conn.cursor() as cur:
            if specific_ticker:
                # Check if ticker exists in sp500_tickers
                cur.execute(f"SELECT ticker FROM {table_name} WHERE ticker = %s;", (specific_ticker,))
                result = cur.fetchone()
                if result:
                    tickers = [specific_ticker]
                else:
                    print(f"❌ Ticker '{specific_ticker}' not found in {table_name} table")
                    return None
            else:
                # Get all tickers
                cur.execute(f"SELECT ticker FROM {table_name} ORDER BY ticker;")
                tickers = [row[0] for row in cur.fetchall()]
        
        print("\n" + "=" * 80)
        if specific_ticker:
            print(f"DOWNLOADING YFINANCE DATA FOR {specific_ticker}")
        else:
            print(f"DOWNLOADING YFINANCE DATA FOR {len(tickers)} TICKERS")
        print("=" * 80)
        
        # Create directory if it doesn't exist
        yfinance_path = os.path.join(DATA_PATH, 'yfinance_validation')
        os.makedirs(yfinance_path, exist_ok=True)
        
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                if len(tickers) > 1:
                    print(f"[{i}/{len(tickers)}] Downloading {ticker}...", end=" ")
                else:
                    print(f"Downloading {ticker}...", end=" ")
                
                # Download data from yfinance - use Ticker object for single ticker
                stock = yf.Ticker(ticker)
                df = stock.history(start='1940-01-01', end=None)
                
                if df.empty:
                    print("❌ No data")
                    failed_tickers.append(ticker)
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Keep only the columns we need
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Save to CSV
                csv_file = os.path.join(yfinance_path, f"{ticker}.csv")
                df.to_csv(csv_file, index=False)
                
                print(f"✅ {len(df)} rows")
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                failed_tickers.append(ticker)
        
        print("=" * 80)
        print(f"✅ Successfully downloaded: {success_count}/{len(tickers)}")
        if failed_tickers:
            print(f"❌ Failed tickers: {', '.join(failed_tickers)}")
        print("=" * 80)
        
        return yfinance_path
        
    except Exception as e:
        print(f"❌ Failed to download yfinance data: {e}")
        return None

def validate_data_quality(conn, table_name):
    """Validate database data against yfinance downloaded CSVs"""
    try:
        yfinance_path = os.path.join(DATA_PATH, 'yfinance_validation')
        
        if not os.path.exists(yfinance_path):
            print("❌ yfinance data not found. Run download_yfinance_data() first.")
            return
        
        # Create mismatches CSV file (overwrite if exists)
        mismatches_file = os.path.join(DATA_PATH, 'validation_mismatches.csv')
        
        # Get last close value per ticker per year from database
        with conn.cursor() as cur:
            cur.execute(f"""
                WITH stock_data AS (
                    SELECT
                        Ticker,
                        Date,
                        Close,
                        EXTRACT(YEAR FROM Date) AS year
                    FROM {table_name}
                    WHERE Ticker IN (SELECT ticker FROM {table_name})
                      AND Close IS NOT NULL
                ),
                ranked_data AS (
                    SELECT
                        Ticker,
                        Date,
                        Close,
                        year,
                        ROW_NUMBER() OVER (
                            PARTITION BY Ticker, year 
                            ORDER BY Date DESC
                        ) AS rn
                    FROM stock_data
                )
                SELECT 
                    Ticker,
                    Date AS last_date,
                    Close AS last_close
                FROM ranked_data
                WHERE rn = 1
                ORDER BY Ticker, last_date;
            """)
            
            db_data = cur.fetchall()
        
        print("\n" + "=" * 80)
        print("DATA QUALITY VALIDATION: Database vs YFinance")
        print("=" * 80)
        
        mismatches = []
        matches = 0
        not_found = 0
        date_not_found = 0
        
        for ticker, date, db_close in db_data:
            csv_file = os.path.join(yfinance_path, f"{ticker}.csv")
            
            if not os.path.exists(csv_file):
                not_found += 1
                continue
            
            try:
                # Read yfinance CSV
                yf_df = pd.read_csv(csv_file)
                
                # Convert Date column - handle timezone if present
                yf_df['Date'] = pd.to_datetime(yf_df['Date'], utc=True)
                # Remove timezone and keep only date
                yf_df['Date'] = yf_df['Date'].dt.tz_localize(None).dt.date
                
                # Convert database date to date object
                date_to_check = date.date() if hasattr(date, 'date') else date
                
                # Find matching row
                matching_rows = yf_df[yf_df['Date'] == date_to_check]
                
                if matching_rows.empty:
                    date_not_found += 1
                    continue
                
                # Get the close value for that date
                yf_close = matching_rows['Close'].values[0]
                
                # Compare values (allow 0.01 difference for rounding)
                if abs(db_close - yf_close) > 0.01:
                    mismatches.append({
                        'ticker': ticker,
                        'date': str(date_to_check),
                        'db_close': db_close,
                        'yf_close': yf_close,
                        'diff': abs(db_close - yf_close)
                    })
                else:
                    matches += 1
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                not_found += 1
                continue
        
        # Save mismatches to CSV (overwrite mode)
        if mismatches:
            mismatches_df = pd.DataFrame(mismatches)
            mismatches_df.to_csv(mismatches_file, index=False)
            print(f"\n💾 Mismatches saved to: {mismatches_file}")
        else:
            # Create empty file with headers if no mismatches
            pd.DataFrame(columns=['ticker', 'date', 'db_close', 'yf_close', 'diff']).to_csv(mismatches_file, index=False)
            print(f"\n💾 No mismatches - empty file created: {mismatches_file}")
        
        print(f"\n{'Status':<20} {'Count':<10}")
        print("-" * 80)
        print(f"{'✅ Matches':<20} {matches:<10}")
        print(f"{'❌ Mismatches':<20} {len(mismatches):<10}")
        print(f"{'⚠️  CSV Not Found':<20} {not_found:<10}")
        print(f"{'⚠️  Date Not Found':<20} {date_not_found:<10}")
        
        if mismatches:
            print("\n" + "=" * 80)
            print("MISMATCHES FOUND:")
            print("=" * 80)
            print(f"{'Ticker':<10} {'Date':<15} {'DB Close':<12} {'YF Close':<12} {'Diff':<10}")
            print("-" * 80)
            for m in mismatches[:50]:  # Show first 50 mismatches
                print(f"{m['ticker']:<10} {m['date']:<15} ${m['db_close']:>10.2f} ${m['yf_close']:>10.2f} ${m['diff']:>8.2f}")
            
            if len(mismatches) > 50:
                print(f"\n... and {len(mismatches) - 50} more mismatches")
        
        print("=" * 80)
        total_checked = matches + len(mismatches)
        accuracy = (matches / total_checked * 100) if total_checked > 0 else 0
        print(f"Data Accuracy: {accuracy:.2f}% ({matches}/{total_checked} records)")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Failed to validate data: {e}")
        import traceback
        traceback.print_exc()

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    start_time = datetime.now()
    
    # Connect to PostgreSQL
    print("\n🔌 Connecting to PostgreSQL...")
    try:
        load_config()

        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to PostgreSQL successfully")
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        return
    
    try:
        if DISPLAY_DATA:
            print("\n📋 Step: Summary")
            display_summary(conn, TABLE_SP500)

        if RUN_QUERY:
            print("\n📋 Step: Running quality control query...")
            run_query(conn, TABLE_SP500)
        
        if DOWNLOAD_YFINANCE:
            print("\n📋 Step: Downloading yfinance data...")
            # Download only AAPL (default)
            #download_yfinance_data(conn, TABLE_SP500)

            # Download specific ticker
            #download_yfinance_data(conn, TABLE_SP500, specific_ticker='A')

            # Download ALL S&P 500 tickers
            download_yfinance_data(conn, TABLE_SP500, specific_ticker=None)
        
        if VALIDATE_DATA:
            print("\n📋 Step 8: Validating data quality...")
            validate_data_quality(conn, TABLE_SP500)
        
        elapsed = datetime.now() - start_time
        print(f"\n⏱️  Total time: {str(elapsed).split('.')[0]}")
        print("\n✅ PROCESS COMPLETED SUCCESSFULLY")
    
    except Exception as e:
        print(f"\n❌ PROCESS FAILED: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()
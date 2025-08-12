import pandas as pd
from datetime import datetime
from tqdm import tqdm
import requests
import csv

# Load the CSV file
df = pd.read_csv("missing_price.csv")

# Headers for Nasdaq API
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'origin': 'https://www.nasdaq.com',
    'priority': 'u=1, i',
    'referer': 'https://www.nasdaq.com/',
    'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
}

# List to hold results
results = []

# Process each symbol/date
for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching close prices"):
    symbol = row['Symbol']
    date_str = row['Date']

    try:
        to_date = datetime.strptime(date_str, "%m/%d/%y")
        from_date = to_date.replace(year=to_date.year - 10)
    except ValueError:
        print(f"Invalid date format: {date_str} for {symbol}")
        continue

    # Format dates
    fromdate_str = from_date.strftime("%Y-%m-%d")
    todate_str = to_date.strftime("%Y-%m-%d")

    # Construct API URL
    url = f"https://api.nasdaq.com/api/quote/{symbol}/historical?assetclass=stocks,etf&fromdate={fromdate_str}&limit=1&todate={todate_str}"

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        print(url)

        rows = data.get("data", {}).get("tradesTable", {}).get("rows", [])
        if rows:
            close_price = rows[0].get("close")
            results.append({
                "Symbol": symbol,
                "Date": todate_str,
                "Close": close_price
            })
        else:
            print(f"No data rows for {symbol} on {todate_str}")

    except Exception as e:
        print(f"Error fetching data for {symbol} on {date_str}: {e}")
    
# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("closing_prices.csv", index=False)
print("✅ Closing prices saved to 'closing_prices.csv'")
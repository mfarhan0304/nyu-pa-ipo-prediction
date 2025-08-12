import requests
import pandas as pd
from datetime import datetime, timedelta
import time

headers = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.nasdaq.com",
    "referer": "https://www.nasdaq.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

def month_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current.strftime("%Y-%m")
        current += timedelta(days=32)
        current = current.replace(day=1)

ipo_data = []

start_year = 2000
end_year = 2024

for year in range(start_year, end_year + 1):
    print(f"\n📅 Scraping IPOs for year {year}...")
    for month in range(1, 13):
        date_str = f"{year}-{month:02d}"
        url = f"https://api.nasdaq.com/api/ipo/calendar?date={date_str}"
        try:
            response = requests.get(url, headers=headers)
            data = response.json()

            priced_rows = data.get("data", {}).get("priced", {}).get("rows", [])
            for row in priced_rows:
                ipo_data.append({
                    "Deal ID": row.get("dealID"),
                    "Symbol": row.get("proposedTickerSymbol"),
                    "Company Name": row.get("companyName"),
                    "Exchange": row.get("proposedExchange"),
                    "Price": row.get("proposedSharePrice"),
                    "Shares": row.get("sharesOffered"),
                    "Date": row.get("pricedDate"),
                    "Offer Amount": row.get("dollarValueOfSharesOffered"),
                    "Status": row.get("dealStatus")
                })

            print(f"  ✅ {date_str}: {len(priced_rows)} IPOs")

        except Exception as e:
            print(f"  ❌ Error scraping {date_str}: {e}")

    time.sleep(1)  # 💤 Sleep after each year to be polite to the server

# Save to CSV
df = pd.DataFrame(ipo_data)
df.to_csv("nasdaq_priced_ipos_2000_2024.csv", index=False)
print(f"\n✅ Done! Total IPOs collected: {len(df)}. Saved to nasdaq_priced_ipos_2000_2024.csv")
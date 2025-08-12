import pandas as pd
import requests
import time
import re
from tqdm import tqdm

# Load initial IPO CSV (must contain a 'DealId' column from first scraping)
df = pd.read_csv("nasdaq_priced_ipos_2000_2024.csv")

# Output list to append enriched records
detailed_data = []

headers = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.nasdaq.com",
    "referer": "https://www.nasdaq.com/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
}

def extract_employee_number(text):
    """Extracts the first number from the employee string."""
    if not text or text == "--":
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None

for i, row in tqdm(df.iterrows(), total=len(df)):
    deal_id = row.get("Deal ID")
    if pd.isna(deal_id):
        continue

    url = f"https://api.nasdaq.com/api/ipo/overview/?dealId={deal_id}"

    try:
        response = requests.get(url, headers=headers)
        data = response.json().get("data", {}).get("poOverview", {})

        enriched = {
            "Symbol": row.get("Symbol"),
            "Company Name": row.get("Company Name"),
            "DealId": deal_id,
            "Employees": extract_employee_number(data.get("NumberOfEmployees", {}).get("value", "")),
            "Total Offering Expense": data.get("TotalExpenseOfTheOffering", {}).get("value", ""),
            # "Shareholder Shares Offered": data.get("ShareholderSharesOffered", {}).get("value", ""),
            "Shares Outstanding": data.get("SharesOutstanding", {}).get("value", ""),
            "Lockup Period (days)": data.get("LockupPeriodNumberofDays", {}).get("value", ""),
            "Lockup Expiration": data.get("LockupPeriodExpirationDate", {}).get("value", ""),
            "Quiet Period Expiration": data.get("QuietPeriodExpirationDate", {}).get("value", ""),
            "CIK": data.get("SECCIK", {}).get("value", "")
        }

        detailed_data.append(enriched)

    except Exception as e:
        print(f"Error fetching dealId {deal_id}: {e}")

    # Sleep every 100 requests
    if (i + 1) % 100 == 0:
        time.sleep(1)

    if (i + 1) % 500 == 0:
        enriched_df = pd.DataFrame(detailed_data)
        enriched_df.to_csv("ipo_details_enriched.csv", mode='a', header=False, index=False)
        detailed_data = []

# Convert to DataFrame and save
enriched_df = pd.DataFrame(detailed_data)
enriched_df.to_csv("ipo_details_enriched.csv", mode='a', header=False, index=False)
print("✅ Enrichment complete. Saved to ipo_details_enriched.csv.")
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.iposcoop.com/2024-pricings/"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Referer": "https://www.iposcoop.com/scoop-track-record-from-2000-to-present/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the IPO table
table = soup.find("table")
rows = table.find_all("tr")

# Extract headers
headers = [th.get_text(strip=True) for th in rows[0].find_all("th")]

# Extract data
data = []
for row in rows[1:]:
    cols = [td.get_text(strip=True) for td in row.find_all("td")]
    if len(cols) == len(headers):
        data.append(dict(zip(headers, cols)))

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("iposcoop_2024_full_table.csv", index=False)
print("Saved as iposcoop_2024_full_table.csv")
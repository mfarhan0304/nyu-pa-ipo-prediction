import pandas as pd
from sec_edgar_downloader import Downloader
from tqdm import tqdm
import time
from datetime import datetime


df = pd.read_csv("missing_cik.csv")
dl = Downloader("NYU", "mxf5233@nyu.edu", "./")


for i, row in tqdm(df.iterrows(), total=len(df)):
    cik = row.get("CIK")
    ipo_date = datetime.strptime(row.get("Date"), "%m/%d/%y")

    try:
        dl.get("F-1", cik, before=ipo_date, limit=1)
    except Exception as e:
        print(f"Error downloading {cik}: {e}")
        continue

    time.sleep(0.2)
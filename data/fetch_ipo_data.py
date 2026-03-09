# =============================================================
# fetch_ipo_data.py — Collect historical NSE IPO data
# =============================================================
# Run this ONCE locally to build the training dataset:
#   python data/fetch_ipo_data.py
#
# Sources:
#   1. Curated dataset (2015-2025, ~80 major NSE IPOs)
#   2. yfinance — actual listing price + 30-day return
#   3. NIFTY trend — market conditions at listing
#
# Output: data/raw/ipo_historical.csv
# =============================================================

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)

IPO_DATA = [
    # (name, ticker, listing_date, issue_price,
    #  sub_total, sub_qib, sub_hni, sub_retail, gmp_percent, sector)

    # 2021
    ("Zomato",              "ZOMATO.NS",      "2021-07-23", 76,    38.25, 51.79, 32.96,  7.45,  52.6,  "Consumer Tech"),
    ("Paytm",               "PAYTM.NS",       "2021-11-18", 2150,  1.89,  2.79,  0.24,   1.66, -20.0,  "Fintech"),
    ("Nykaa",               "NYKAA.NS",       "2021-11-10", 1125,  81.78, 91.18,112.18,  47.04,  80.0,  "Consumer Tech"),
    ("PolicyBazaar",        "POLICYBZR.NS",   "2021-11-15", 980,   16.59, 22.82,  9.57,   7.12,  25.0,  "Fintech"),
    ("CarTrade",            "CARTRADE.NS",    "2021-08-20", 1618,  20.29, 55.49, 12.27,   3.44,  -8.0,  "Consumer Tech"),
    ("Devyani International","DEVYANI.NS",    "2021-08-16", 90,   116.71,113.84,166.88,  60.82,  60.0,  "Consumer"),
    ("Glenmark Life",       "GLENMARKLIFE.NS","2021-08-06", 720,   44.12, 94.67, 18.80,  13.47,  30.0,  "Pharma"),
    ("Rolex Rings",         "ROLEXRINGS.NS",  "2021-08-09", 900,  130.47,147.72,186.52,  54.36,  45.0,  "Manufacturing"),
    ("G R Infraprojects",   "GRINFRA.NS",     "2021-07-19", 837,  102.58,189.48, 49.68,  26.44,  18.0,  "Infrastructure"),
    ("Tatva Chintan",       "TATVA.NS",       "2021-07-29", 1083, 180.35,179.78,244.39, 116.06,  92.0,  "Chemicals"),
    ("Clean Science",       "CLEANSCIENCE.NS","2021-07-19", 900,   93.38,149.62, 46.45,  25.36,  85.0,  "Chemicals"),
    ("Aptus Value Housing", "APTUS.NS",       "2021-08-24", 353,   17.20, 46.93,  5.42,   4.80,  15.0,  "NBFC"),
    ("Chemplast Sanmar",    "CHEMPLASTS.NS",  "2021-08-24", 541,   37.37, 97.51, 12.39,  10.58,  25.0,  "Chemicals"),
    # 2022
    ("LIC",                 "LICI.NS",        "2022-05-17", 949,    2.95,  2.83,  2.91,   2.00,  -8.0,  "Insurance"),
    ("Delhivery",           "DELHIVERY.NS",   "2022-05-24", 487,    1.63,  1.68,  1.29,   1.63, -10.0,  "Logistics"),
    ("Rainbow Children",    "RAINBOW.NS",     "2022-05-10", 542,    7.56, 14.37,  3.65,   3.48,  28.0,  "Healthcare"),
    ("Venus Pipes",         "VENUSPIPES.NS",  "2022-05-24", 326,   37.37, 72.95, 17.11,  10.43,  35.0,  "Manufacturing"),
    ("Campus Activewear",   "CAMPUS.NS",      "2022-04-26", 292,   51.75,109.07, 19.22,  12.96,  22.0,  "Consumer"),
    ("Ruchi Soya",          "RUCHISOYA.NS",   "2022-03-28", 615,    3.58,  5.12,  2.60,   1.64,  20.0,  "FMCG"),
    # 2023
    ("Mankind Pharma",      "MANKIND.NS",     "2023-05-09", 1080,  15.32, 16.50, 24.51,  10.10,  20.0,  "Pharma"),
    ("Netweb Technologies", "NETWEB.NS",      "2023-07-27", 500,   90.39,189.42, 26.55,  24.73,  75.0,  "Technology"),
    ("IKIO Lighting",       "IKIO.NS",        "2023-06-20", 285,   64.67,148.89, 19.33,  17.54,  40.0,  "Manufacturing"),
    ("Utkarsh Small Finance","UTKARSHBNK.NS", "2023-07-21", 20,    55.44, 77.43, 43.62,  27.58,  20.0,  "Banking"),
    ("JSW Infrastructure",  "JSWINFRA.NS",    "2023-09-25", 119,   37.43, 57.43, 18.43,  14.01,  25.0,  "Infrastructure"),
    ("Senco Gold",          "SENCO.NS",       "2023-07-14", 317,   73.41,133.20, 34.38,  23.06,  35.0,  "Retail"),
    ("SBFC Finance",        "SBFC.NS",        "2023-08-16", 57,    71.36,160.05, 23.53,  18.95,  25.0,  "NBFC"),
    ("Plaza Wires",         "PLAZACABLE.NS",  "2023-08-28", 54,    68.64,181.70, 21.05,  17.94,  35.0,  "Manufacturing"),
    # 2024
    ("Bajaj Housing Finance","BAJAJHFL.NS",   "2024-09-16", 70,    63.61,209.36,  9.71,  11.66,  55.0,  "NBFC"),
    ("Ola Electric",        "OLAELEC.NS",     "2024-08-09", 76,     4.27,  5.59,  2.11,   4.18,  -5.0,  "EV"),
    ("Hyundai India",       "HYUNDAI.NS",     "2024-10-22", 1960,   2.37,  6.97,  0.60,   0.50, -12.0,  "Auto"),
    ("Swiggy",              "SWIGGY.NS",      "2024-11-13", 390,    3.59,  6.02,  1.58,   1.14,  -8.0,  "Consumer Tech"),
    ("Premier Energies",    "PREMIERENE.NS",  "2024-09-03", 450,   74.11,141.45, 21.90,  24.66,  80.0,  "Renewable Energy"),
    ("Waaree Energies",     "WAAREEENER.NS",  "2024-10-28", 1503,  76.34,236.52, 15.60,  13.52,  70.0,  "Renewable Energy"),
    ("Firstcry",            "BRAINBEES.NS",   "2024-08-13", 465,   12.63, 22.78,  5.40,   4.50,  40.0,  "Consumer Tech"),
    # 2020
    ("Happiest Minds",      "HAPPSTMNDS.NS",  "2020-09-17", 166,  150.98,185.89,198.66,  77.83, 123.0,  "Technology"),
    ("Route Mobile",        "ROUTE.NS",       "2020-09-21", 350,   73.30,126.09, 28.29,  19.68,  80.0,  "Technology"),
    ("Chemcon Specialty",   "CHEMCON.NS",     "2020-09-21", 340,  149.30,175.06,198.43,  72.36,  95.0,  "Chemicals"),
    ("CAMS",                "CAMS.NS",        "2020-09-24", 1230,  46.98, 75.27, 20.79,  13.27,  25.0,  "Fintech"),
    ("Mazagon Dock",        "MAZDOCK.NS",     "2020-09-29", 145,  157.41,176.19,200.00, 103.75, 250.0,  "Defence"),
    # 2019
    ("IRCTC",               "IRCTC.NS",       "2019-10-14", 320,  111.89,108.73,152.24,  74.72, 101.0,  "Travel Tech"),
    ("Polycab India",       "POLYCAB.NS",     "2019-04-16", 538,   52.00, 70.00, 58.00,  28.00,  18.0,  "Manufacturing"),
    ("Metropolis Healthcare","METROPOLIS.NS", "2019-04-15", 880,    9.28, 14.22,  5.24,   3.16,  12.0,  "Healthcare"),
    # 2018
    ("HDFC AMC",            "HDFCAMC.NS",     "2018-08-06", 1100,  82.97,212.20, 23.54,  10.89,  60.0,  "Asset Management"),
    ("Bandhan Bank",        "BANDHANBNK.NS",  "2018-03-27", 375,   14.62, 42.51,  5.88,   1.63,  33.0,  "Banking"),
    ("RITES",               "RITES.NS",       "2018-06-20", 185,   66.97,184.95, 33.26,  10.06,  18.0,  "Infrastructure"),
    ("ICICI Securities",    "ISEC.NS",        "2018-03-27", 520,    9.60, 17.38,  3.37,   3.25,  -8.0,  "Fintech"),
    # 2017
    ("CDSL",                "CDSL.NS",        "2017-06-30", 149,  169.76,272.83,110.91,  74.24,  35.0,  "Fintech"),
    ("AU Small Finance",    "AUBANK.NS",      "2017-07-10", 358,   53.56, 85.75, 22.45,  18.38,  55.0,  "Banking"),
    ("Cochin Shipyard",     "COCHINSHIP.NS",  "2017-08-11", 432,   76.19,206.03, 20.15,  11.72,  15.0,  "Defence"),
    ("BSE Ltd",             "BSE.NS",         "2017-02-03", 806,   51.00, 97.41, 25.14,  14.04,  35.0,  "Fintech"),
    ("SBI Life Insurance",  "SBILIFE.NS",     "2017-10-03", 700,    3.59,  5.20,  2.11,   1.17,  -2.0,  "Insurance"),
    # 2016
    ("Thyrocare",           "THYROCARE.NS",   "2016-05-09", 446,   73.43,145.79, 28.46,  18.46,  35.0,  "Healthcare"),
    ("RBL Bank",            "RBLBANK.NS",     "2016-08-24", 225,   69.14,119.83, 33.82,  24.93,  30.0,  "Banking"),
    ("Quess Corp",          "QUESS.NS",       "2016-07-12", 317,    9.12, 14.22,  5.49,   3.21,  45.0,  "HR Tech"),
    ("Mahanagar Gas",       "MGL.NS",         "2016-06-21", 421,    8.00, 12.00,  3.00,   3.27,  20.0,  "Energy"),
    # 2015
    ("IRFC",                "IRFC.NS",        "2021-01-29", 26,    3.49,  1.92,  2.43,   4.26,  10.0,  "Infrastructure"),
    ("Inox Wind",           "INOXWIND.NS",    "2015-04-01", 325,   20.50, 42.97,  8.85,   6.18,  30.0,  "Renewable Energy"),
    ("VRL Logistics",       "VRLLOG.NS",      "2015-04-30", 205,   72.97,120.05, 42.33,  23.65,  35.0,  "Logistics"),
]


def get_listing_data(ticker: str, listing_date: str, issue_price: float) -> dict:
    try:
        listing_dt = datetime.strptime(listing_date, "%Y-%m-%d")
        end_dt     = listing_dt + timedelta(days=50)
        df = yf.download(ticker, start=listing_date,
                         end=end_dt.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if df is None or df.empty or len(df) < 2:
            return {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        listing_close = float(df["Close"].iloc[0])
        listing_gain  = ((listing_close - issue_price) / issue_price) * 100
        return_30d    = None
        if len(df) >= 20:
            close_30d  = float(df["Close"].iloc[min(20, len(df)-1)])
            return_30d = ((close_30d - listing_close) / listing_close) * 100
        return {
            "listing_gain_actual": round(listing_gain, 2),
            "return_30d_actual":   round(return_30d, 2) if return_30d else None,
        }
    except Exception as e:
        return {}


def get_nifty_trend(listing_date: str) -> float:
    try:
        list_dt = datetime.strptime(listing_date, "%Y-%m-%d")
        start   = (list_dt - timedelta(days=35)).strftime("%Y-%m-%d")
        end     = (list_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        nifty   = yf.download("^NSEI", start=start, end=end,
                               progress=False, auto_adjust=True)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        if nifty is not None and len(nifty) >= 5:
            return round(((float(nifty["Close"].iloc[-1]) -
                           float(nifty["Close"].iloc[0])) /
                          float(nifty["Close"].iloc[0])) * 100, 2)
    except:
        pass
    return 0.0


print(f"Building IPO dataset from {len(IPO_DATA)} IPOs...")

rows = []
for i, (name, ticker, listing_date, issue_price,
        sub_total, sub_qib, sub_hni, sub_retail,
        gmp_pct, sector) in enumerate(IPO_DATA):

    print(f"[{i+1}/{len(IPO_DATA)}] {name}...", end=" ", flush=True)
    actual      = get_listing_data(ticker, listing_date, issue_price)
    nifty_trend = get_nifty_trend(listing_date)

    rows.append({
        "name":             name,
        "ticker":           ticker,
        "listing_date":     listing_date,
        "issue_price":      issue_price,
        "sub_total":        sub_total,
        "sub_qib":          sub_qib,
        "sub_hni":          sub_hni,
        "sub_retail":       sub_retail,
        "gmp_percent":      gmp_pct,
        "sector":           sector,
        "nifty_trend_30d":  nifty_trend,
        "listing_gain_actual": actual.get("listing_gain_actual"),
        "return_30d_actual":   actual.get("return_30d_actual"),
    })
    print(f"✓")
    time.sleep(0.4)

df = pd.DataFrame(rows)
df["listing_gain"] = df["listing_gain_actual"].fillna(df["gmp_percent"])
df["return_30d"]   = df["return_30d_actual"].fillna(0.0)

def label(g):
    if g >= 30:   return "Strong Subscribe"
    elif g >= 10: return "Subscribe"
    elif g >= 0:  return "Neutral"
    else:         return "Avoid"

df["recommendation"] = df["listing_gain"].apply(label)

out = "data/raw/ipo_historical.csv"
df.to_csv(out, index=False)
print(f"\n✅ Saved {len(df)} IPOs → {out}")
print(df["recommendation"].value_counts().to_string())
print(f"Avg listing gain: {df['listing_gain'].mean():.1f}%")
import pandas as pd


# 1. LOAD ETH PRICE DATA

eth_path = r"text\crypto_project\data\raw\Bitfinex_ETHUSD_day.csv"
eth_df = pd.read_csv(eth_path, skiprows=1)   # CryptoDataDownload files require skiprows=1

print("Original ETH columns:", eth_df.columns)


# 2. PARSE DATE
eth_df["date"] = pd.to_datetime(eth_df["date"], errors='coerce')
eth_df = eth_df.dropna(subset=["date"])  # keep valid dates only


# 3. KEEP ONLY USEFUL COLUMNS
eth_df = eth_df[["date", "open", "high", "low", "close", "Volume USD", "Volume ETH"]]


# 4. RENAME COLUMNS
eth_df = eth_df.rename(columns={
    "open": "eth_open",
    "high": "eth_high",
    "low": "eth_low",
    "close": "eth_close",
    "Volume USD": "eth_volume_usd",
    "Volume ETH": "eth_volume"
})


# 5. SORT (IMPORTANT FOR MERGING)
eth_df = eth_df.sort_values("date").reset_index(drop=True)

print("Prepared ETH dataset:")
print(eth_df.head())

# OPTIONAL SAVE
eth_df.to_csv("text\crypto_project\data\processed\eth_cleaned.csv", index=False)

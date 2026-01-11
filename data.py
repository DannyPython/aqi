import pandas as pd
import requests
import os
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"

# Change this line to switch cities for your entire project
CURRENT_FILE_PATH = r'PM 2.5 Data\hanoi-air-quality.csv'

# Mapping files to API City Names
LOCATIONS = {
    # Singapore
    r'PM 2.5 Data\south,-singapore-air-quality.csv': 'singapore/south',
    r'PM 2.5 Data\north,-singapore-air-quality.csv': 'singapore/north',
    r'PM 2.5 Data\east,-singapore-air-quality.csv':  'singapore/east',
    r'PM 2.5 Data\west,-singapore-air-quality.csv':  'singapore/west',
    r'PM 2.5 Data\central,-singapore-air-quality.csv': 'singapore/central',

    # Vietnam
    r'PM 2.5 Data\hanoi-air-quality.csv': 'hanoi',
    r'PM 2.5 Data\da-nang-air-quality.csv': 'danang',

    # India
    r'PM 2.5 Data\mundka,-delhi, delhi, india-air-quality.csv': 'delhi/mundka',
}

# ==========================================
# 2. INTERNAL API FUNCTIONS (Hidden from Main)
# ==========================================
def _fetch_and_update(csv_path):
    """
    Connects to WAQI API and appends today's data to the CSV.
    This is internal (starts with _) because main.py doesn't need to call it directly.
    """
    # 1. Identify City
    # Normalize paths to handle Windows/Mac slashes
    norm_path = os.path.normpath(csv_path)
    # create a lookup dict with normalized keys
    norm_locations = {os.path.normpath(k): v for k, v in LOCATIONS.items()}
    
    if norm_path not in norm_locations:
        print(f" WARNING: No API mapping found for {csv_path}. Skipping update.")
        return

    city_name = norm_locations[norm_path]
    url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"
    
    print(f"🔄 Checking for new data for: {city_name}...")

    try:
        response = requests.get(url, timeout=10)
        payload = response.json()
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return

    if payload.get('status') != 'ok':
        print(f"❌ API Error: {payload.get('data')}")
        return

    # 2. Extract Data
    data = payload.get('data', {})
    iaqi = data.get('iaqi', {})
    time_info = data.get('time', {})
    
    date_str = time_info.get('s', '').split(' ')[0] # YYYY-MM-DD

    if not date_str:
        return

    new_row = {
        'date': date_str,
        ' pm25': iaqi.get('pm25', {}).get('v', ''),
        ' pm10': iaqi.get('pm10', {}).get('v', ''),
        ' o3':   iaqi.get('o3', {}).get('v', ''),
        ' so2':  iaqi.get('so2', {}).get('v', ''),
        ' co':   iaqi.get('co', {}).get('v', '')
    }

    # 3. Update CSV safely
    try:
        # If file exists, check for duplicates
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if date_str in df['date'].values:
                print(f"✅ Data up-to-date ({date_str} exists).")
                return
            else:
                # Append
                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(csv_path, index=False)
                print(f"✅ UPDATED: Added data for {date_str}")
        else:
            # Create new file
            df = pd.DataFrame([new_row])
            df.to_csv(csv_path, index=False)
            print(f"✅ CREATED: New file for {city_name}")

    except Exception as e:
        print(f"❌ File Error: {e}")

# ==========================================
# 3. PUBLIC DATA LOADER (Called by Main)
# ==========================================
def get_data(file_path=CURRENT_FILE_PATH):
    """
    The only function main.py needs to know.
    1. Updates data from API
    2. Loads CSV
    3. Cleans and formats it
    """
    
    # A. Try to update first
    _fetch_and_update(file_path)

    # B. Load the file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    # C. Clean Columns (The "Space" Bug Fix)
    df.columns = df.columns.str.strip()

    # D. Convert to Numeric
    cols = ['pm25', 'pm10', 'o3', 'so2', 'co']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # E. Sort Dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # F. Create Lag
    if 'pm25' in df.columns:
        df['PM2.5_Lag1'] = df['pm25'].shift(1)

    df_clean = df.dropna()
    print(f"📊 Loaded {len(df_clean)} rows for model.")
    return df_clean

if __name__ == "__main__":
    # Test run if executed directly
    df = get_data()
    print(df.tail())

import pandas as pd
import os

CSV_FILES = [
    'PM 2.5 Data/hanoi-air-quality.csv',
    'PM 2.5 Data/south,-singapore-air-quality.csv',
    'PM 2.5 Data/north,-singapore-air-quality.csv',
    'PM 2.5 Data/east,-singapore-air-quality.csv',
    'PM 2.5 Data/west,-singapore-air-quality.csv',
    'PM 2.5 Data/central,-singapore-air-quality.csv',
    'PM 2.5 Data/da-nang-air-quality.csv',
    'PM 2.5 Data/mundka,-delhi, delhi, india-air-quality.csv',
]

for path in CSV_FILES:
    if not os.path.exists(path):
        print(f"⚠️  Skipping (not found): {path}")
        continue

    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        before = len(df)
        df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
        df = df.sort_values('date')
        df['date'] = df['date'].dt.strftime('%d/%m/%Y')

        df.to_csv(path, index=False)
        print(f"✅ Sorted {before} rows → {path}")

    except Exception as e:
        print(f"❌ Error with {path}: {e}")

print("\nDone!")

import pandas as pd
df = pd.read_csv('data/raw/content_refresh_anonymized.csv')
print('Total rows:', len(df))

# Grain check
grain = df.groupby('content_id').size().reset_index(name='count')
print('Grain holds (all 1 row per content_id):', grain['count'].nunique() == 1)

# Missingness
for col in ['search_volume', 'word_count']:
    overall = df[col].isna().mean()
    print(f'{col} overall missing fraction: {overall:.3f}')

# Trailing 90d window
print('impressions_90d min:', df['impressions_90d'].min())
print('impressions_90d max:', df['impressions_90d'].max())
print('rows with impressions_90d = 0:', (df['impressions_90d'] == 0).sum())

# Missingness by content_type
for col in ['search_volume', 'word_count']:
    by_type = df.groupby('content_type')[col].apply(lambda x: x.isna().mean()).round(3)
    print(f'{col} missing by content_type:')
    print(by_type)
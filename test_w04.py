import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/content_refresh_anonymized.csv')
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)
df['has_clicks'] = (df['clicks_90d'] > 0).astype(int)

# Signal 1
has_clicks_decline = df[df['has_clicks'] == 1]['is_declining_label'].mean()
no_clicks_decline = df[df['has_clicks'] == 0]['is_declining_label'].mean()
v1 = 'CONFIRMED' if no_clicks_decline > has_clicks_decline else 'OPPOSITE' if no_clicks_decline < has_clicks_decline else 'MIXED'
print('Signal 1 - Decline rate with clicks:', round(has_clicks_decline, 4))
print('Signal 1 - Decline rate without clicks:', round(no_clicks_decline, 4))
print('Signal 1 Verdict:', v1)

print()

# Signal 2
median_er = df['engagement_rate'].median()
high_er_decline = df[df['engagement_rate'] > median_er]['is_declining_label'].mean()
low_er_decline = df[df['engagement_rate'] <= median_er]['is_declining_label'].mean()
v2 = 'CONFIRMED' if high_er_decline < low_er_decline else 'OPPOSITE' if high_er_decline > low_er_decline else 'MIXED'
print('Signal 2 - Decline rate high engagement:', round(high_er_decline, 4))
print('Signal 2 - Decline rate low engagement:', round(low_er_decline, 4))
print('Signal 2 Verdict:', v2)

print()

# Signal 3
df['ai_bucket'] = pd.cut(df['ai_traffic_pct'], bins=[-1, 20, 40, 60, 100, 101], labels=['0-20','20-40','40-60','60-100','100+'])
ai_decline = df.groupby('ai_bucket')['is_declining_label'].mean().round(4)
v3 = 'FALSE' if ai_decline.std() < 0.05 else 'MIXED'
print('Signal 3 - Decline rate by AI traffic %:')
print(ai_decline)
print('Signal 3 Verdict:', v3)

print()

# Flag-linked test
ft_decline = df[df['freshness_tier'] == 'never']['is_declining_label'].mean()
other_decline = df[df['freshness_tier'] != 'never']['is_declining_label'].mean()
print('Flag-linked test - Decline rate never updated:', round(ft_decline, 4))
print('Flag-linked test - Decline rate had update:', round(other_decline, 4))
if ft_decline > other_decline:
    print('Interpretation: stale content associated with higher decline (confirms stale = decline assumption)')
else:
    print('Interpretation: stale content NOT associated with higher decline (assumption not supported)')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/raw/content_refresh_anonymized.csv')

# Compute the label from trend_direction (as the prep step does)
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)

engineered = df.copy()

# Drop IDs - they are pseudonyms for grouping only, never features
engineered = engineered.drop(['content_id', 'client_id'], axis=1)

# engineered features
engineered['has_keyword_data'] = engineered['search_volume'].notna().astype(int)
engineered['has_word_count'] = engineered['word_count'].notna().astype(int)
engineered['has_clicks'] = (engineered['clicks_90d'] > 0).astype(int)
engineered['has_ai_sessions'] = (engineered['ai_sessions_90d'] > 0).astype(int)
engineered['log_impressions_90d'] = np.log1p(engineered['impressions_90d'])
engineered['log_clicks_90d'] = np.log1p(engineered['clicks_90d'])
engineered['log_sessions_90d'] = np.log1p(engineered['sessions_90d'])
engineered['log_ai_sessions_90d'] = np.log1p(engineered['ai_sessions_90d'])

for cat_col in ['content_type', 'main_intent', 'competition_level', 'age_tier', 
    'freshness_tier', 'impression_tier', 'position_tier']:
    dummies = pd.get_dummies(engineered[cat_col], prefix=cat_col, drop_first=True)
    engineered = engineered.join(dummies)

# Ensure only numeric columns for the model
X_clean = engineered.drop(['is_declining_label', 'trend_direction', 'trend_pct'], axis=1, errors='ignore')
X_clean = X_clean.select_dtypes(include=[np.number])
y = engineered['is_declining_label']

scores_rf = cross_val_score(RandomForestClassifier(random_state=42), X_clean.fillna(0), y, cv=5, scoring='accuracy')
print('Random split RF (no leaky cols) accuracy mean:', scores_rf.mean(), 'std:', scores_rf.std())

# Window alignment check
print('Correlation impressions_last_30d <-> impressions_prev_30d:', df['impressions_last_30d'].corr(df['impressions_prev_30d']))

# Base rate
print('Positive class rate (is_declining_label mean):', y.mean())

# Rate gotcha
print('ctr range:', df['ctr'].min(), 'to', df['ctr'].max())
print('rows where ctr > 1 (would be 100%+ if misinterpreted):', (df['ctr'] > 1).sum())
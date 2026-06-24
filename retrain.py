import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('data/Cleaned_aqi.csv')

# Features and target
X = df[['City','PM2.5','PM10','NO','NO2','NOx',
        'NH3','CO','SO2','O3','Benzene',
        'Toluene','Xylene','year','month','day','day_of_week']]

y = df['AQI_Bucket']

# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Save models
joblib.dump(model,   'model/aqi_model.pkl')
joblib.dump(encoder, 'model/label_encoder.pkl')

print("✅ Model retrained successfully!")
print("Classes:", list(encoder.classes_))
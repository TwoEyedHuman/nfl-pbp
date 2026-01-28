from data_sourcing import load_training_data
from sklearn.ensemble import GradientBoostingClassifier
import joblib

X, y = load_training_data()

wp_model = GradientBoostingClassifier()
wp_model.fit(X, y)

print("Model Trained!")

joblib.dump(wp_model, 'models/xgboost_wp.pkl')
from data_sourcing import load_training_data
from sklearn.linear_model import LogisticRegression
import joblib

X, y = load_training_data()

wp_model = LogisticRegression()
wp_model.fit(X, y)

print("Model Trained!")
joblib.dump(wp_model, 'models/logistic_regression_wp.pkl')
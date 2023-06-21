import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("fraud_dataset.csv")

# Preprocess the data
# Perform data cleaning, feature engineering, and scaling if required

# Split the dataset into training and testing sets
X = data.drop("fraud_label", axis=1)
y = data["fraud_label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling to address class imbalance
oversampler = SMOTE()
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Perform feature selection to focus on the most informative features
selector = SelectKBest(f_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Scale the features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Fraud Detection System Performance:")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# Additional advanced features:
# - Cross-validation for more robust evaluation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation scores: ", cv_scores)

# - Hyperparameter tuning to optimize model performance
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

# - Ensemble methods like bagging or boosting for improved accuracy
bagging_model = BaggingClassifier(base_estimator=model, n_estimators=10)
bagging_model.fit(X_train, y_train)
bagging_accuracy = bagging_model.score(X_test, y_test)
print("Bagging accuracy: ", bagging_accuracy)

adaboost_model = AdaBoostClassifier(base_estimator=model, n_estimators=50)
adaboost_model.fit(X_train, y_train)
adaboost_accuracy = adaboost_model.score(X_test, y_test)
print("AdaBoost accuracy: ", adaboost_accuracy)

# - Anomaly detection techniques to identify unusual patterns
isolation_forest = IsolationForest()
isolation_forest.fit(X_train, y_train)
anomaly_predictions = isolation_forest.predict(X_test)

# - Feature importance analysis to understand the most relevant features
feature_importances = model.feature_importances_
print("Feature Importances: ", feature_importances)

# - Model interpretability techniques like SHAP values or feature importance plots
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")
plt.show()

# Continuously update and improve the model based on new fraud patterns and techniques.

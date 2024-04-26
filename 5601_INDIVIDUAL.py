from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler  # Import the RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Import pandas for data manipulation if needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
data = pd.read_csv('training.csv')

# Handling missing values
for col in data.columns:
    if data[col].dtype == "object":
        # For categorical columns use mode
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        # For numerical columns use mean
        data[col] = data[col].fillna(data[col].mean())

# Converting categorical variables to numeric codes
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target variable
X = data[['CurrentEquipmentDays','MonthlyMinutes','MonthlyRevenue','MonthsInService','UnansweredCalls',
          'OutboundCalls','InboundCalls','AgeHH1']]
y = data['Churn']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize the RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train, y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
Y_pred_rf = rf_model.predict(X_test)

# Metrics for Random Forest
accuracy_rf = accuracy_score(y_test, Y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, Y_pred_rf)
class_report_rf = classification_report(y_test, Y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
# Metrics for Random Forest
print(f"Accuracy: {accuracy_rf}")
print(f"Confusion Matrix:\n{conf_matrix_rf}")
print(f"Classification Report:\n{class_report_rf}")
print(f"ROC AUC Score: {roc_auc_rf}")

# Get feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Plot the mean decrease in accuracy
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Mean Decrease in Accuracy")
plt.bar(range(X_train.shape[1]), feature_importances[indices],
       color="b", align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel("Feature")
plt.ylabel("Mean Decrease in Accuracy")
plt.tight_layout()
plt.show()



import joblib

# After training the rf_model...
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(y_test, 'y_test.joblib')
joblib.dump(rf_model.predict_proba(X_test), 'probabilities.joblib')


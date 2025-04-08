import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_X.csv')

# Separate features and target
X = train_df.drop(['PatientID', 'HeartDisease'], axis=1)
y = train_df['HeartDisease']
X_test = test_df.drop(['PatientID'], axis=1)

# Identify categorical and numerical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipelines for numerical and categorical data
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Define the classifier with fixed parameters (for run time efficiency)
base_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Create a pipeline that transforms data then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', base_model)
])

# Hyperparameter grid
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7],
    'classifier__n_estimators': [100, 150, 200, 250, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'classifier__gamma': [0, 0.1, 0.2, 0.3],
    'classifier__min_child_weight': [1, 3, 5]
}

# Set up Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Macro F1 scorer
f1_scorer = make_scorer(f1_score, average='macro')

# GridSearchCV on the full training set
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best CV F1 score on full training:", grid_search.best_score_)

# Calibrate the best estimator using cross-validation (using 5-fold CV)
calibrated_clf = CalibratedClassifierCV(grid_search.best_estimator_, cv=5)
calibrated_clf.fit(X, y)

# Tune threshold using cross-validated predictions on the full training set
# This gives a more robust estimate of the optimal threshold
val_probs = cross_val_predict(calibrated_clf, X, y, cv=5, method='predict_proba')[:, 1]

thresholds = np.linspace(0.1, 0.9, 81)
best_thresh = 0.5
best_f1 = 0.0
for thresh in thresholds:
    preds_thresh = (val_probs >= thresh).astype(int)
    current_f1 = f1_score(y, preds_thresh, average='macro')
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = thresh

print(f"Optimal threshold after calibration: {best_thresh:.2f} with F1: {best_f1:.4f}")

# Get predicted probabilities on the test set using the calibrated classifier
test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
test_predictions = (test_probs >= best_thresh).astype(int)

# Prepare submission file
submission = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'HeartDisease': test_predictions
})
submission.to_csv('y_predict.csv', index=False)
print("Submission file 'y_predict.csv' created successfully.")

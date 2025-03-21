import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
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

# Define the LightGBM classifier
lgbm_model = LGBMClassifier(random_state=42)

# Create a pipeline that transforms data then fits the model
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

# Expanded hyperparameter grid for LightGBM
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7],
    'classifier__n_estimators': [100, 150, 200, 250, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'classifier__num_leaves': [15, 31, 63],
    'classifier__min_child_samples': [10, 20, 30],
    'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'classifier__reg_alpha': [0, 0.01, 0.1],
    'classifier__reg_lambda': [0, 0.01, 0.1]
}

# Set up Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Macro F1 scorer for evaluation
f1_scorer = make_scorer(f1_score, average='macro')

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(lgbm_pipeline, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print("Best parameters (LightGBM):", grid_search.best_params_)
print("Best F1 score on CV (LightGBM):", grid_search.best_score_)

# Generate predictions on the test set using the best LightGBM estimator
predictions = grid_search.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'HeartDisease': predictions
})
submission.to_csv('y_predict.csv', index=False)

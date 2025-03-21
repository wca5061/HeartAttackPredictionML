import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Define the classifier; removing use_label_encoder as it's not used in recent versions.
model = XGBClassifier(eval_metric='logloss', random_state=42)

# Create a pipeline that transforms data then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Expanded hyperparameter grid for fine tuning
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7],
    'classifier__n_estimators': [100, 150, 200, 250, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'classifier__gamma': [0, 0.1, 0.2, 0.3],
    'classifier__min_child_weight': [1, 3, 5]
}

# Set up Stratified K-Fold cross-validation to maintain class balance
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Macro F1 scorer for imbalanced metrics
f1_scorer = make_scorer(f1_score, average='macro')

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=f1_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best F1 score on CV:", grid_search.best_score_)

# Generate predictions on the test set using the best estimator
predictions = grid_search.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'PatientID': test_df['PatientID'],
    'HeartDisease': predictions
})
submission.to_csv('y_predict.csv', index=False)

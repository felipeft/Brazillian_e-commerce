import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ..features.features import create_features, create_preprocessor

def train_model():
    """
    Carrega os dados, treina o modelo e o salva.
    """
    df = pd.read_parquet('data/processed/orders_table.parquet')

    processed_df = create_features(df)
    
    X = processed_df.drop('is_late', axis=1)
    y = processed_df['is_late']
    
    preprocessor = create_preprocessor()
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    joblib.dump(model_pipeline, 'models/logistic_regression_model.pkl')
    print("Modelo treinado e salvo com sucesso em models/logistic_regression_model.pkl")

if __name__ == '__main__':
    train_model()
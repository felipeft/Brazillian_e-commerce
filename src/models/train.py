import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from ..features.features import create_features_and_target, create_preprocessor

def train_model():
    """
    Carrega os dados, treina o modelo com a melhor estratégia (SMOTE + Random Forest)
    e o salva.
    """
    try:
        df = pd.read_parquet('data/processed/orders_table.parquet')
    except FileNotFoundError:
        print("Erro: O arquivo 'data/processed/orders_table.parquet.parquet' não foi encontrado. Por favor, execute a etapa de ETL e pré-processamento primeiro.")
        return

    # 1. Engenharia de Features e seleção
    processed_df = create_features_and_target(df)
    
    X = processed_df.drop('is_late', axis=1)
    y = processed_df['is_late']
    
    # 2. Separar dados de treino e teste
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Pré-processamento e Modelagem com Pipeline Robusto
    preprocessor = create_preprocessor()
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    model_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])
    
    # Treinar o modelo
    model_pipeline.fit(X_train, y_train)

    # 4. Salvar o modelo treinado
    joblib.dump(model_pipeline, 'models/logistic_regression_model.pkl')
    print("Modelo treinado e salvo com sucesso em models/logistic_regression_model.pkl")

if __name__ == '__main__':
    train_model()
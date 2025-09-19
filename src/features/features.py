# src/features.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo e seleciona as features para o modelo.
    """
    # Converter as colunas de data para o tipo datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])

    # Calcular a diferença de dias entre a entrega real e a estimada
    df['delivery_diff'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

    # Criação da variável 'is_late'
    df['is_late'] = df['delivery_diff'] > 0
    df['is_late'] = df['is_late'].astype(int)

    # AQUI ESTÁ A PRIMEIRA MUDANÇA: 'order_item_id' foi removido
    features = [
        'customer_state', 'customer_city', 'product_category_name',
        'price', 'freight_value'
    ]
    
    target = 'is_late'

    # Removendo linhas com valores nulos que seriam problemáticos para o modelo
    df_cleaned = df.dropna(subset=features + [target])

    return df_cleaned[features + [target]]

def create_preprocessor():
    """
    Cria e retorna o pré-processador de dados (ColumnTransformer)
    com a etapa de imputação.
    """
    # AQUI ESTÁ A SEGUNDA MUDANÇA: 'order_item_id' foi removido
    numeric_features = ['price', 'freight_value']
    categorical_features = ['customer_state', 'customer_city', 'product_category_name']

    # Criando pipelines para cada tipo de coluna
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Criando o pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 

def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável alvo e seleciona features mais relevantes.
    """
    # Converter as colunas de data para o tipo datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
    df['order_delivered_carrier_date'] = pd.to_datetime(df['order_delivered_carrier_date'])


    # Calcular a diferença de dias entre a entrega real e a estimada (nosso target)
    df['delivery_diff'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    df['is_late'] = (df['delivery_diff'] > 0).astype(int)

    # Calcular o tempo de aprovação do pagamento em dias
    df['payment_approval_time_days'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / (60*60*24)
    # Calcular o tempo de entrega para a transportadora em dias
    df['carrier_delivery_time_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / (60*60*24)

    # Features que fazem mais sentido para a previsão
    selected_features = [
        'customer_state',               
        'product_category_name_english',  
        'price',                        
        'freight_value',                
        'payment_installments',         
        'review_score',                 
        'seller_state',                 
        'product_weight_g',             
        'payment_approval_time_days',   
        'carrier_delivery_time_days'    
    ]

    target = 'is_late'
    
    # Remover linhas com valores nulos nas features selecionadas
    df_cleaned = df.dropna(subset=selected_features + [target])

    return df_cleaned[selected_features + [target]]

def create_preprocessor():
    """
    Cria e retorna o pré-processador de dados (ColumnTransformer)
    """
    numeric_features = ['price', 'freight_value', 'payment_installments', 'review_score', 'product_weight_g', 'payment_approval_time_days', 'carrier_delivery_time_days']
    categorical_features = ['customer_state', 'product_category_name_english', 'seller_state']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
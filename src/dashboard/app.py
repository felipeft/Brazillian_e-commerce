import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o modelo treinado de forma robusta e independente
# O comando os.getcwd() retorna o diretório de trabalho atual
project_root = os.getcwd()
model_path = os.path.join(project_root, 'models', 'logistic_regression_model.pkl')

try:
    model = joblib.load(model_path)
    st.success("Modelo carregado com sucesso!")
except FileNotFoundError:
    st.error(f"Erro: O arquivo do modelo não foi encontrado em {model_path}. Por favor, execute 'python -m src.models.train' no terminal para treinar e salvar o modelo.")
    st.stop()

# --- Título e Descrição ---
st.title('Analisador de Vendas da Olist')
st.markdown('### Previsão de Entrega de Pedidos')
st.markdown("""
Este aplicativo prevê se um pedido será entregue no prazo ou com atraso,
com base em algumas características do pedido, do cliente e do produto.
""")

# --- Criar a barra lateral para entrada de dados ---
st.sidebar.header('Entrada de Dados do Pedido')

# Dados sobre o cliente e o produto
customer_state = st.sidebar.selectbox('Estado do Cliente', options=['SP', 'RJ', 'MG', 'DF', 'BA', 'SC', 'PR', 'RS', 'CE', 'GO', 'PE', 'ES', 'PR', 'PA', 'MT', 'MS', 'MA', 'RN', 'PB', 'AL', 'SE', 'RO', 'PI', 'RR', 'AM', 'AC', 'TO'])
seller_state = st.sidebar.selectbox('Estado do Vendedor', options=['SP', 'RJ', 'MG', 'DF', 'BA', 'SC', 'PR', 'RS', 'CE', 'GO', 'PE', 'ES', 'PR', 'PA', 'MT', 'MS', 'MA', 'RN', 'PB', 'AL', 'SE', 'RO', 'PI', 'RR', 'AM', 'AC', 'TO'])
product_category_name_english = st.sidebar.selectbox('Categoria do Produto', options=['furniture_decor', 'computers_accessories', 'health_beauty', 'bed_bath_table', 'sports_leisure', 'garden_tools', 'toys', 'auto', 'cool_stuff', 'musical_instruments', 'telephony', 'housewares', 'books_general_interest', 'fashion_bags_accessories', 'pet_shop', 'watches_gifts', 'electronics', 'baby'])

# Dados numéricos
price = st.sidebar.number_input('Preço do Produto (R$)', min_value=0.0, value=50.0)
freight_value = st.sidebar.number_input('Valor do Frete (R$)', min_value=0.0, value=10.0)
payment_installments = st.sidebar.number_input('Número de Parcelas', min_value=1, value=1)
review_score = st.sidebar.number_input('Score de Avaliação (1-5)', min_value=1, max_value=5, value=5)
product_weight_g = st.sidebar.number_input('Peso do Produto (g)', min_value=0, value=500)
payment_approval_time_days = st.sidebar.number_input('Tempo de Aprovação de Pagamento (dias)', min_value=0.0, value=0.5)
carrier_delivery_time_days = st.sidebar.number_input('Tempo de Envio para a Transportadora (dias)', min_value=0.0, value=1.0)

# Botão para fazer a previsão
if st.button('Fazer Previsão'):
    # Preparar os dados de entrada em um DataFrame
    input_data = pd.DataFrame([{
        'customer_state': customer_state,
        'product_category_name_english': product_category_name_english,
        'price': price,
        'freight_value': freight_value,
        'payment_installments': payment_installments,
        'review_score': review_score,
        'seller_state': seller_state,
        'product_weight_g': product_weight_g,
        'payment_approval_time_days': payment_approval_time_days,
        'carrier_delivery_time_days': carrier_delivery_time_days
    }])
    
    # Fazer a previsão
    try:
        prediction = model.predict(input_data)[0]
        
        # Exibir o resultado
        if prediction == 1:
            st.error('O modelo prevê que o pedido será **ATRASADO**.')
        else:
            st.success('O modelo prevê que o pedido será **ENTREGUE NO PRAZO**.')
            
    except Exception as e:
        st.error(f"Ocorreu um erro durante a previsão: {e}")
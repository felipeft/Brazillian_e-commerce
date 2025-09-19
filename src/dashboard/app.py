import os
import streamlit as st
import pandas as pd
import joblib

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

# --- Criar a barra lateral para entrada de dados ---
st.sidebar.header('Entrada de Dados do Pedido')

# Removemos a entrada 'order_item_id'
price = st.sidebar.number_input('Preço do Produto (R$)', min_value=0.0, value=50.0)
freight_value = st.sidebar.number_input('Valor do Frete (R$)', min_value=0.0, value=10.0)

# Simplificando as opções para demonstração
customer_state = st.sidebar.selectbox(
    'Estado do Cliente',
    options=['SP', 'RJ', 'MG', 'DF', 'BA', 'SC', 'PR', 'RS']
)

customer_city = st.sidebar.text_input('Cidade do Cliente', value='sao paulo')

product_category_name = st.sidebar.selectbox(
    'Categoria do Produto',
    options=['beleza_saude', 'cama_mesa_banho', 'esporte_lazer', 'moveis_decoracao', 'informatica_acessorios']
)

# Botão para fazer a previsão
if st.button('Fazer Previsão'):
    # Preparar os dados de entrada em um DataFrame
    input_data = pd.DataFrame([{
        'price': price,
        'freight_value': freight_value,
        'customer_state': customer_state,
        'customer_city': customer_city,
        'product_category_name': product_category_name
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
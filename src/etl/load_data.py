import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/olist")

def load_csv(file_name, parse_dates=None):
    path = DATA_DIR / file_name
    df = pd.read_csv(path, parse_dates=parse_dates)
    print(f"✅ {file_name}: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(df.head(), "\n")
    return df

def main():
    # Ler CSVs principais com nomes corretos
    orders = load_csv("olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    
    order_items = load_csv("olist_order_items_dataset.csv")
    payments = load_csv("olist_order_payments_dataset.csv")
    reviews = load_csv("olist_order_reviews_dataset.csv", parse_dates=[
        "review_creation_date","review_answer_timestamp"])
    customers = load_csv("olist_customers_dataset.csv")
    products = load_csv("olist_products_dataset.csv")
    sellers = load_csv("olist_sellers_dataset.csv")
    geolocation = load_csv("olist_geolocation_dataset.csv")
    product_cat_trans = load_csv("product_category_name_translation.csv")

    print("✅ Todos os CSVs carregados com sucesso!")

if __name__ == "__main__":
    main()

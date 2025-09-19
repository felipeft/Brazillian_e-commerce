import pandas as pd
from pathlib import Path

# Caminhos
RAW_DIR = Path("../../data/olist")
PROCESSED_DIR = Path("../../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # --- Carregar datasets ---
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp",
                                      "order_approved_at",
                                      "order_delivered_carrier_date",
                                      "order_delivered_customer_date",
                                      "order_estimated_delivery_date"])
    
    order_items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv",
                          parse_dates=["review_creation_date",
                                       "review_answer_timestamp"])
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    products = pd.read_csv(RAW_DIR / "olist_products_dataset.csv")
    sellers = pd.read_csv(RAW_DIR / "olist_sellers_dataset.csv")
    product_cat_trans = pd.read_csv(RAW_DIR / "product_category_name_translation.csv")

    # --- Merge: customers + orders ---
    df = orders.merge(customers, on="customer_id", how="left")

    # --- Merge: order_items ---
    df = df.merge(order_items, on="order_id", how="left")

    # --- Merge: payments ---
    df = df.merge(payments, on="order_id", how="left")

    # --- Merge: reviews ---
    df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

    # --- Merge: products ---
    df = df.merge(products, on="product_id", how="left")

    # --- Merge: sellers ---
    df = df.merge(sellers, on="seller_id", how="left")

    # --- Merge: tradução categorias ---
    df = df.merge(product_cat_trans, on="product_category_name", how="left")

    # --- Criar features básicas ---
    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["estimated_time_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay"] = (df["delivery_time_days"] > df["estimated_time_days"]).astype(int)
    df["total_price"] = df["price"] + df["freight_value"]

    # --- Salvar ---
    output_path = PROCESSED_DIR / "orders_table.parquet"
    df.to_parquet(output_path, index=False)
    print(f"✅ Tabela final salva em: {output_path}")
    print(df.head())

if __name__ == "__main__":
    main()

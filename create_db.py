import pandas as pd
import sqlite3

# Cargar los CSV con delimitador personalizado
products = pd.read_csv("data/products.csv", delimiter=";")  # o sep=";"
orders = pd.read_csv("data/orders.csv", delimiter=";")

# Conectar a una base SQLite (archivo o memoria)
con = sqlite3.connect("ecomarket.db")

# Guardar las tablas
products.to_sql("products", con, if_exists="replace", index=False)
orders.to_sql("orders", con, if_exists="replace", index=False)

con.close()

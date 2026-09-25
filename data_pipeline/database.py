
import sqlite3
import pandas as pd

CSV_PATH = "data_pipeline/books.csv"
DB_PATH = "data_pipeline/books.db"

df = pd.read_csv(CSV_PATH)

connection = sqlite3.connect(DB_PATH)

df.to_sql("books", connection, if_exists="replace", index=False)

query = """
SELECT category, COUNT(*) AS total_books,
       ROUND(AVG(price_gbp), 2) AS average_price_gbp
FROM books
GROUP BY category
ORDER BY category;
"""

print(pd.read_sql_query(query, connection))

connection.close()
print("Database created successfully!")
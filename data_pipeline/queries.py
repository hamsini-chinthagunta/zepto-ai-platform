
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).with_name("books.db")
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")

def table_exists(name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)
    ).fetchone() is not None

def columns(table):
    return [
        row[1]
        for row in conn.execute(f'PRAGMA table_info("{table}")')
    ]

# 1. Normalize the existing flat books table if needed.
book_cols = columns("books")

if "category_id" not in book_cols:
    print("Converting existing books table to normalized tables...")

    conn.execute("BEGIN")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    conn.execute("""
        INSERT OR IGNORE INTO categories (category_name)
        SELECT DISTINCT category
        FROM books
        WHERE category IS NOT NULL
    """)

    conn.execute("""
        CREATE TABLE books_normalized (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price TEXT,
            star_rating INTEGER,
            availability INTEGER,
            price_gbp REAL,
            price_inr REAL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    conn.execute("""
        INSERT INTO books_normalized
            (title, price, star_rating, availability,
             price_gbp, price_inr, category_id)
        SELECT
            b.title,
            b.price,
            b.star_rating,
            b.availability,
            b.price_gbp,
            b.price_inr,
            c.category_id
        FROM books b
        JOIN categories c
          ON b.category = c.category_name
    """)

    conn.execute("DROP TABLE books")
    conn.execute("ALTER TABLE books_normalized RENAME TO books")
    conn.commit()

else:
    print("Books table is already normalized.")

print("\nTABLES:")
print(pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    conn
).to_string(index=False))

print("\nBOOKS TABLE COLUMNS:")
print(columns("books"))

print("\nCATEGORIES TABLE COLUMNS:")
print(columns("categories"))

# 2. SELECT + WHERE
# availability is stored as an integer: 1 = in stock.
query1 = """
SELECT title, price_gbp, star_rating, availability
FROM books
WHERE availability = 1
LIMIT 10;
"""
print("\n1) SELECT + WHERE: In-stock books")
print(pd.read_sql_query(query1, conn).to_string(index=False))

# 3. ORDER BY
query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""
print("\n2) ORDER BY: 10 most expensive books")
print(pd.read_sql_query(query2, conn).to_string(index=False))

# 4. LIMIT
query3 = """
SELECT title, price_gbp
FROM books
LIMIT 5;
"""
print("\n3) LIMIT: First 5 books")
print(pd.read_sql_query(query3, conn).to_string(index=False))

# 5. DISTINCT
query4 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""
print("\n4) DISTINCT: All categories")
print(pd.read_sql_query(query4, conn).to_string(index=False))

# 6. IN
query5 = """
SELECT b.title, c.category_name
FROM books b
JOIN categories c
  ON b.category_id = c.category_id
WHERE c.category_name IN ('Travel', 'Mystery')
ORDER BY c.category_name, b.title
LIMIT 15;
"""
print("\n5) IN: Books in Travel or Mystery")
print(pd.read_sql_query(query5, conn).to_string(index=False))

# 7. BETWEEN
query6 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 10 AND 25
ORDER BY price_gbp;
"""
print("\n6) BETWEEN: Books priced from £10 to £25")
print(pd.read_sql_query(query6, conn).to_string(index=False))

# 8. JOIN using pd.read_sql_query
join_sql = """
SELECT
    b.title,
    b.price_gbp,
    c.category_name
FROM books b
JOIN categories c
  ON b.category_id = c.category_id
ORDER BY c.category_name, b.title;
"""
sql_join_df = pd.read_sql_query(join_sql, conn)
sql_join_df.columns = ["title", "price_gbp", "category"]

print("\n7) JOIN using pd.read_sql_query:")
print(sql_join_df.head(10).to_string(index=False))

# 9. Same JOIN using pandas.merge
books_df = pd.read_sql_query("SELECT * FROM books", conn)
categories_df = pd.read_sql_query("SELECT * FROM categories", conn)

pandas_join_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)[["title", "price_gbp", "category_name"]]

pandas_join_df.columns = ["title", "price_gbp", "category"]
pandas_join_df = pandas_join_df.sort_values(
    ["category", "title"]
).reset_index(drop=True)

print("\nSame JOIN using pd.merge:")
print(pandas_join_df.head(10).to_string(index=False))

# Compare full results, not just the displayed first 10 rows.
sql_sorted = sql_join_df.sort_values(
    ["category", "title"]
).reset_index(drop=True)

print("\nDo SQL JOIN and pandas merge match?")
print(sql_sorted.equals(pandas_join_df))

# 10. Foreign-key check
print("\nForeign key check (empty result means no violations):")
print(conn.execute("PRAGMA foreign_key_check").fetchall())

conn.close()
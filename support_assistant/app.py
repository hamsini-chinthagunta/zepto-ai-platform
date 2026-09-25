
import sqlite3

DB_PATH = "data_pipeline/books.db"


def answer(question):
    q = question.lower().strip()

    # Open database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        # Greetings: match whole input only, so "highest" won't trigger "hi"
        if q in ["hi", "hii", "hiii", "hello", "hey", "good morning", "good evening"]:
            result = "Hi! Ask me about book counts, categories, prices, ratings, or stock."

        # Total number of books
        elif any(phrase in q for phrase in [
            "how many books", "total books", "number of books",
            "books in the dataset", "dataset size"
        ]):
            cur.execute("SELECT COUNT(*) FROM books")
            count = cur.fetchone()[0]
            result = f"There are {count} books in the dataset."

        # Number of categories
        elif "how many categories" in q or "number of categories" in q:
            cur.execute("SELECT COUNT(DISTINCT category) FROM books")
            count = cur.fetchone()[0]
            result = f"There are {count} categories in the dataset."

        # List categories
        elif "which categories" in q or "categories available" in q or "list categories" in q:
            cur.execute("SELECT DISTINCT category FROM books ORDER BY category")
            categories = [row[0] for row in cur.fetchall()]
            result = "Available categories: " + ", ".join(categories)

        # Highest average price by category
        elif "highest average price" in q or "category costs the most on average" in q:
            cur.execute("""
                SELECT category, ROUND(AVG(price_gbp), 2) AS avg_price
                FROM books
                GROUP BY category
                ORDER BY avg_price DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            result = f"{row[0]} has the highest average price: £{row[1]}."

        # Lowest average price by category
        elif "lowest average price" in q or "category costs the least on average" in q:
            cur.execute("""
                SELECT category, ROUND(AVG(price_gbp), 2) AS avg_price
                FROM books
                GROUP BY category
                ORDER BY avg_price ASC
                LIMIT 1
            """)
            row = cur.fetchone()
            result = f"{row[0]} has the lowest average price: £{row[1]}."

        # Average price of all books
        elif "average price" in q or "mean price" in q:
            cur.execute("SELECT ROUND(AVG(price_gbp), 2) FROM books")
            avg_price = cur.fetchone()[0]
            result = f"Average price: £{avg_price}."

        # Cheapest book
        elif "cheapest book" in q or "lowest price book" in q:
            cur.execute("""
                SELECT title, price_gbp
                FROM books
                ORDER BY price_gbp ASC
                LIMIT 1
            """)
            title, price = cur.fetchone()
            result = f"The cheapest book is '{title}' at £{price}."

        # Most expensive book
        elif "most expensive book" in q or "highest price book" in q:
            cur.execute("""
                SELECT title, price_gbp
                FROM books
                ORDER BY price_gbp DESC
                LIMIT 1
            """)
            title, price = cur.fetchone()
            result = f"The most expensive book is '{title}' at £{price}."

        # Top 5 highest-rated books
        elif "top 5" in q and ("rated" in q or "rating" in q):
            cur.execute("""
                SELECT title, star_rating
                FROM books
                ORDER BY star_rating DESC, title ASC
                LIMIT 5
            """)
            rows = cur.fetchall()
            result = "Top 5 highest-rated books:\n" + "\n".join(
                f"{i}. {title} — {rating}/5"
                for i, (title, rating) in enumerate(rows, start=1)
            )

        # Average rating
        elif "average rating" in q or "mean rating" in q:
            cur.execute("SELECT ROUND(AVG(star_rating), 2) FROM books")
            avg_rating = cur.fetchone()[0]
            result = f"Average rating: {avg_rating} out of 5."

        # Highest rating
        elif "highest rating" in q or "highest-rated" in q or "top rated" in q:
            cur.execute("""
                SELECT title, star_rating
                FROM books
                ORDER BY star_rating DESC, title ASC
                LIMIT 1
            """)
            title, rating = cur.fetchone()
            result = f"One of the highest-rated books is '{title}', rated {rating}/5."

        # Stock availability
        elif "all books in stock" in q or "are all books in stock" in q:
            cur.execute("SELECT COUNT(*) FROM books WHERE availability = 1")
            in_stock = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM books")
            total = cur.fetchone()[0]

            if in_stock == total:
                result = f"Yes, all {total} books are in stock."
            else:
                result = f"{in_stock} out of {total} books are in stock."

        elif "stock" in q or "in stock" in q or "availability" in q:
            cur.execute("SELECT COUNT(*) FROM books WHERE availability = 1")
            in_stock = cur.fetchone()[0]
            result = f"{in_stock} books are in stock."

        else:
            result = (
                "I can answer questions about:\n"
                "- Total books and categories\n"
                "- Average price and category price comparisons\n"
                "- Cheapest and most expensive books\n"
                "- Ratings and top 5 highest-rated books\n"
                "- Stock availability"
            )

        return result

    finally:
        conn.close()


print("Book Support Assistant")
print("Type 'exit' to quit.")

while True:
    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit", "bye"]:
        print("Assistant: Goodbye!")
        break

    if not question:
        continue

    print("Assistant:", answer(question))
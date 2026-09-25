
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def scrape_category(category_name, category_url):
    records = []
    url = category_url

    while url:
        soup = get_soup(url)

        for book in soup.select("article.product_pod"):
            title_tag = book.select_one("h3 a")
            price_tag = book.select_one(".price_color")
            rating_tag = book.select_one(".star-rating")
            availability_tag = book.select_one(".availability")

            if not all([title_tag, price_tag, rating_tag, availability_tag]):
                continue

            records.append({
                "title": title_tag.get("title", title_tag.get_text(strip=True)),
                "price": price_tag.get_text(strip=True),
                "star_rating": rating_tag.get("class", ["", "Unknown"])[1],
                "availability": availability_tag.get_text(" ", strip=True),
                "category": category_name
            })

        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None

    return records

def main():
    soup = get_soup(BASE_URL)
    category_links = soup.select(".side_categories ul li ul li a")

    all_books = []

    for link in category_links[:3]:
        name = link.get_text(strip=True)
        category_url = urljoin(BASE_URL, link["href"])
        print(f"Scraping {name}...")
        all_books.extend(scrape_category(name, category_url))

    df = pd.DataFrame(all_books)
    df = df.drop_duplicates(subset=["title"])

    df["price_gbp"] = (
        df["price"].str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .astype(float)
    )

    rating_map = {
        "One": 1, "Two": 2, "Three": 3,
        "Four": 4, "Five": 5
    }
    df["star_rating"] = df["star_rating"].map(rating_map)

    df["availability"] = df["availability"].str.contains(
        "In stock", case=False, na=False
    )

    # Illustrative conversion rate; replace with the rate specified by your assignment.
    GBP_TO_INR = 110.0
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df.to_csv("data_pipeline/books.csv", index=False)
    print("\nScraping completed!")
    print("Total books:", len(df))
    print("Categories:", df["category"].nunique())
    print(df.head())

if __name__ == "__main__":
    main()

# Zepto AI Platform

A book data collection and analytics project with a simple book support assistant.

## Project Features

- Scrapes book information from Books to Scrape.
- Collects book titles, prices, ratings, stock availability, and categories.
- Cleans and transforms the scraped data.
- Converts prices from GBP to INR using an illustrative exchange rate.
- Stores the data in an SQLite database.
- Uses Pandas for summary statistics and category analysis.
- Generates a chart of average book prices by category.
- Provides a rule-based support assistant that answers questions using the database.

## Project Structure

```text
zepto-ai-platform/
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── books.csv
│   └── books.db
├── analytics/
│   ├── analysis.py
│   └── books_by_category.png
├── support_assistant/
│   └── app.py
├── .gitignore
└── README.md
```

## How to Run

Install dependencies:

```bash
python -m pip install requests beautifulsoup4 pandas matplotlib
```

Run the scraper:

```bash
python data_pipeline/scraper.py
```

Create the database:

```bash
python data_pipeline/database.py
```

Run the analysis:

```bash
python analytics/analysis.py
```

Start the support assistant:

```bash
python support_assistant/app.py
```

## Data Source

Books to Scrape: https://books.toscrape.com/

## Note

The support assistant is rule-based and uses predefined question patterns with SQLite queries. The GBP-to-INR conversion rate in the scraper is illustrative and should be replaced with the rate specified by the project requirements, if applicable.
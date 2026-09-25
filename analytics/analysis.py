
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_pipeline/books.csv")

print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

print("\nSummary statistics:")
print(df[["price_gbp", "price_inr", "star_rating"]].describe())

print("\nAverage price by category:")
print(df.groupby("category")["price_gbp"].mean().round(2))

print("\nAverage rating by category:")
print(df.groupby("category")["star_rating"].mean().round(2))

category_counts = df["category"].value_counts()
category_counts.plot(kind="bar", title="Books by Category")
plt.xlabel("Category")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig("analytics/books_by_category.png")
plt.close()

print("\nChart saved to analytics/books_by_category.png")
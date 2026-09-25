from pathlib import Path
import re
import sqlite3

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DB_PATH = OUTPUT_DIR / "books.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0)"
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a page and return its parsed HTML."""
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_rating(rating_text: str):
    """Convert One/Five etc. into an integer rating."""
    mapping = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    return mapping.get(rating_text)


def parse_price(price_text: str):
    """Convert a GBP price string to float."""
    match = re.search(r"[\d.]+", price_text)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def parse_stock(availability_text: str):
    """Convert availability text into a boolean."""
    text = availability_text.strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


def scrape_category(category_url: str, category_name: str) -> list:
    """Scrape all books from one category."""
    records = []
    url = category_url

    while url:
        soup = get_soup(url)

        for article in soup.select("article.product_pod"):
            title_element = article.select_one("h3 a")
            price_element = article.select_one(".price_color")
            availability_element = article.select_one(".availability")
            rating_element = article.select_one("p.star-rating")

            title = title_element.get("title", "").strip()
            price_text = price_element.get_text(strip=True)
            availability_text = availability_element.get_text(" ", strip=True)

            rating_classes = rating_element.get("class", [])
            rating_text = next(
                (
                    value
                    for value in rating_classes
                    if value in {"One", "Two", "Three", "Four", "Five"}
                ),
                None,
            )

            records.append(
                {
                    "title": title,
                    "price_text": price_text,
                    "star_rating": rating_text,
                    "availability_text": availability_text,
                    "category": category_name,
                }
            )

        next_link = soup.select_one("li.next a")

        if next_link:
            next_href = next_link.get("href")
            url = requests.compat.urljoin(url, next_href)
        else:
            url = None

    return records


def get_categories() -> list:
    """Get category names and URLs from the catalogue."""
    soup = get_soup(BASE_URL)

    categories = []

    for link in soup.select(".side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        href = link.get("href")

        if href:
            categories.append(
                {
                    "name": name,
                    "url": requests.compat.urljoin(BASE_URL, href),
                }
            )

    return categories


def clean_data(raw_records: list) -> pd.DataFrame:
    """Clean and transform scraped records."""
    df = pd.DataFrame(raw_records)

    df["price_gbp"] = df["price_text"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability_text"].apply(parse_stock)

    # Numeric parsing failures are median-imputed as required.
    for column in ["price_gbp", "rating"]:
        if df[column].isna().any():
            median_value = df[column].median()

            if pd.isna(median_value):
                raise ValueError(
                    f"Cannot median-impute column '{column}' because "
                    "all values are missing."
                )

            df[column] = df[column].fillna(median_value)

    # Boolean parsing failures are dropped because there is no sensible
    # numeric median for a boolean availability field.
    df = df.dropna(subset=["in_stock"]).copy()

    df["rating"] = df["rating"].round().astype(int)
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["in_stock"] = df["in_stock"].astype(bool)

    # Required fixed project conversion rate.
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]


def create_database(df: pd.DataFrame):
    """Create normalized SQLite database and load cleaned records."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            );
            """
        )

        categories_df = (
            df[["category"]]
            .drop_duplicates()
            .rename(columns={"category": "category_name"})
        )

        categories_df.to_sql(
            "categories",
            connection,
            if_exists="append",
            index=False,
        )

        category_map = pd.read_sql(
            "SELECT category_id, category_name FROM categories",
            connection,
        )

        books_df = df.merge(
            category_map,
            left_on="category",
            right_on="category_name",
            how="left",
        )

        books_df = books_df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_id",
            ]
        ]

        books_df["in_stock"] = books_df["in_stock"].astype(int)

        books_df.to_sql(
            "books",
            connection,
            if_exists="append",
            index=False,
        )

        connection.commit()

    finally:
        connection.close()


def main():
    print("Discovering categories...")
    categories = get_categories()

    # The first three categories satisfy the requirement of scraping
    # at least three categories.
    selected_categories = categories[:3]

    print("Selected categories:")
    for category in selected_categories:
        print(f"  - {category['name']}")

    raw_records = []

    for category in selected_categories:
        print(f"Scraping: {category['name']}")
        records = scrape_category(
            category["url"],
            category["name"],
        )
        raw_records.extend(records)

    print(f"Raw rows scraped: {len(raw_records)}")

    df = clean_data(raw_records)

    if len(df) < 60:
        raise RuntimeError(
            f"Only {len(df)} valid books were produced. "
            "The assignment requires at least 60."
        )

    if df["category"].nunique() < 3:
        raise RuntimeError(
            "The assignment requires at least 3 different categories."
        )

    print(f"Clean rows: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")

    create_database(df)

    print(f"Database created at: {DB_PATH}")


if __name__ == "__main__":
    main()

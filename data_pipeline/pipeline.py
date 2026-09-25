import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
FIXED_GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "books.db"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "query_outputs.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Zepto-Data-Pipeline/1.0)"
}

STAR_RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# ============================================================
# HTTP HELPERS
# ============================================================

def get_page(url):
    """Download a webpage and raise an error for HTTP failures."""
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()
    return response.text


# ============================================================
# SCRAPING
# ============================================================

def scrape_category(category_url, category_name):
    """
    Scrape all books from one category, including pagination.
    """

    records = []
    current_url = category_url
    page_number = 1

    while current_url:

        print(
            f"Scraping category='{category_name}', "
            f"page={page_number}"
        )

        html = get_page(current_url)
        soup = BeautifulSoup(html, "html.parser")

        products = soup.select("article.product_pod")

        if not products:
            print("No products found on page. Stopping pagination.")
            break

        for product in products:

            # ------------------------------------------------
            # Title
            # ------------------------------------------------
            title_element = product.select_one("h3 a")

            title = ""

            if title_element:
                title = (
                    title_element.get("title")
                    or title_element.get_text(strip=True)
                )

            # ------------------------------------------------
            # Price
            # ------------------------------------------------
            price_element = product.select_one(".price_color")

            price_text = ""

            if price_element:
                price_text = price_element.get_text(" ", strip=True)

            # ------------------------------------------------
            # Rating
            # ------------------------------------------------
            rating_element = product.select_one(
                "p.star-rating"
            )

            star_rating = ""

            if rating_element:
                classes = rating_element.get("class", [])

                for value in classes:
                    if value in STAR_RATING_MAP:
                        star_rating = value
                        break

            # ------------------------------------------------
            # Availability
            # ------------------------------------------------
            availability_element = product.select_one(
                ".availability"
            )

            availability = ""

            if availability_element:
                availability = availability_element.get_text(
                    " ",
                    strip=True
                )

            records.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name,
                }
            )

        # ----------------------------------------------------
        # Pagination
        # ----------------------------------------------------
        next_button = soup.select_one(
            "li.next a"
        )

        if next_button and next_button.get("href"):

            next_href = next_button["href"]

            if next_href.startswith("http"):
                current_url = next_href
            else:
                current_url = requests.compat.urljoin(
                    current_url,
                    next_href
                )

            page_number += 1

        else:
            current_url = None

    return records


def scrape_books():
    """
    Scrape at least three categories.

    Three complete categories easily provide more than
    the required 60 books.
    """

    categories = [
        (
            "Travel",
            "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
        ),
        (
            "Mystery",
            "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"
        ),
        (
            "Historical Fiction",
            "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"
        ),
    ]

    all_records = []

    for category_name, category_url in categories:

        records = scrape_category(
            category_url,
            category_name
        )

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    print("\nRaw scraping complete.")
    print(f"Total scraped rows: {len(df)}")
    print(
        "Categories:",
        df["category"].nunique()
    )

    return df


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def parse_price(value):
    """
    Extract a numeric GBP price from text.

    Examples:
        £51.77 -> 51.77
        GBP 51.77 -> 51.77
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    # Remove HTML entities/spaces and search for number.
    match = re.search(
        r"(\d+(?:\.\d+)?)",
        text.replace(",", "")
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_rating(value):
    """
    Convert textual rating into integer 1-5.

    Handles values such as:
        One
        Two
        Three
        Four
        Five

    Also handles strings containing the rating word.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    for word, number in STAR_RATING_MAP.items():

        if re.search(
            rf"\b{re.escape(word)}\b",
            text,
            flags=re.IGNORECASE
        ):
            return number

    # Fallback for numeric rating text.
    match = re.search(
        r"\b([1-5])\b",
        text
    )

    if match:
        return int(match.group(1))

    return None


def parse_stock(value):
    """
    Convert availability text into boolean.

    'In stock' -> True
    Other values -> False
    """

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return "in stock" in text


def clean_data(df):
    """
    Clean raw scraped data.

    Numeric parsing failures are handled using median
    imputation. Rows with missing/unusable text fields
    such as title/category are dropped because those fields
    are required identifiers/descriptors.
    """

    cleaned = df.copy()

    print("\n========== RAW DATA ==========")
    print(cleaned.head())
    print(cleaned.info())

    # --------------------------------------------------------
    # Required text fields
    # --------------------------------------------------------

    cleaned["title"] = (
        cleaned["title"]
        .astype("string")
        .str.strip()
    )

    cleaned["category"] = (
        cleaned["category"]
        .astype("string")
        .str.strip()
    )

    # Empty strings -> NA
    cleaned["title"] = cleaned["title"].replace(
        "",
        pd.NA
    )

    cleaned["category"] = cleaned["category"].replace(
        "",
        pd.NA
    )

    # Drop rows where essential descriptive fields are missing.
    before = len(cleaned)

    cleaned = cleaned.dropna(
        subset=["title", "category"]
    )

    dropped = before - len(cleaned)

    if dropped:
        print(
            f"Dropped {dropped} rows with missing "
            "title/category."
        )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    cleaned["price_gbp"] = cleaned["price"].apply(
        parse_price
    )

    price_invalid = cleaned["price_gbp"].isna().sum()

    if price_invalid > 0:

        median_price = cleaned["price_gbp"].median()

        if pd.isna(median_price):
            raise ValueError(
                "Unable to calculate median price. "
                "No valid price values were scraped."
            )

        print(
            f"Imputing {price_invalid} invalid/missing "
            f"price values with median = {median_price:.2f}"
        )

        cleaned["price_gbp"] = (
            cleaned["price_gbp"]
            .fillna(median_price)
        )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    cleaned["rating"] = cleaned["star_rating"].apply(
        parse_rating
    )

    rating_invalid = cleaned["rating"].isna().sum()

    if rating_invalid > 0:

        median_rating = cleaned["rating"].median()

        if pd.isna(median_rating):
            raise ValueError(
                "Unable to calculate median rating. "
                "No valid rating values were scraped."
            )

        # Rating must remain an integer from 1 to 5.
        median_rating = int(round(median_rating))

        print(
            f"Imputing {rating_invalid} invalid/missing "
            f"rating values with median = {median_rating}"
        )

        cleaned["rating"] = (
            cleaned["rating"]
            .fillna(median_rating)
        )

    cleaned["rating"] = (
        cleaned["rating"]
        .round()
        .astype("int64")
    )

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    cleaned["in_stock"] = cleaned["availability"].apply(
        parse_stock
    )

    cleaned["in_stock"] = cleaned["in_stock"].astype(bool)

    # --------------------------------------------------------
    # INR conversion
    # --------------------------------------------------------

    cleaned["price_gbp"] = cleaned[
        "price_gbp"
    ].astype(float)

    cleaned["price_inr"] = (
        cleaned["price_gbp"]
        * FIXED_GBP_TO_INR
    ).round(2)

    # --------------------------------------------------------
    # Final column selection
    # --------------------------------------------------------

    cleaned = cleaned[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ].reset_index(drop=True)

    print("\n========== CLEANED DATA ==========")
    print(cleaned.head())

    print("\nData types:")
    print(cleaned.dtypes)

    print(
        f"\nFinal cleaned rows: {len(cleaned)}"
    )

    print(
        f"Final categories: "
        f"{cleaned['category'].nunique()}"
    )

    if len(cleaned) < 60:
        raise ValueError(
            f"Only {len(cleaned)} rows available. "
            "At least 60 rows are required."
        )

    if cleaned["category"].nunique() < 3:
        raise ValueError(
            "At least 3 categories are required."
        )

    return cleaned


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(df):
    """
    Create normalized SQLite database.

    Tables:
        categories
        books

    Relationship:
        categories.category_id
            |
            |--- books.category_id
    """

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    # Enable foreign-key constraints.
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Categories table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    # --------------------------------------------------------
    # Books table
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # --------------------------------------------------------
    # Insert categories
    # --------------------------------------------------------

    categories = sorted(
        df["category"]
        .dropna()
        .unique()
        .tolist()
    )

    cursor.executemany(
        """
        INSERT INTO categories(category_name)
        VALUES (?)
        """,
        [(category,) for category in categories]
    )

    # --------------------------------------------------------
    # Category mapping
    # --------------------------------------------------------

    category_rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        """
    ).fetchall()

    category_map = {
        name: category_id
        for category_id, name in category_rows
    }

    # --------------------------------------------------------
    # Insert books
    # --------------------------------------------------------

    rows = []

    for _, row in df.iterrows():

        rows.append(
            (
                str(row["title"]),
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                category_map[row["category"]],
            )
        )

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows
    )

    connection.commit()

    print(
        f"\nSQLite database created: {DB_PATH}"
    )

    print(
        f"Inserted {len(rows)} books."
    )

    return connection


# ============================================================
# SQL QUERIES
# ============================================================

def run_sql_queries(connection):
    """
    Execute required SQL queries and save their output.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    queries = {

        "Query 1 - SELECT and WHERE":
        """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE price_gbp > 20
        ORDER BY price_gbp DESC;
        """,

        "Query 2 - ORDER BY and LIMIT":
        """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """,

        "Query 3 - DISTINCT":
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
        """,

        "Query 4 - IN":
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE category_id IN (
            SELECT category_id
            FROM categories
            WHERE category_name IN (
                'Travel',
                'Mystery'
            )
        )
        ORDER BY rating DESC;
        """,

        "Query 5 - BETWEEN":
        """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
        """,

        "Query 6 - JOIN":
        """
        SELECT
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.price_gbp DESC
        LIMIT 10;
        """
    }

    output_blocks = []

    print("\n========== SQL QUERY RESULTS ==========")

    for name, query in queries.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        print(query.strip())

        result = pd.read_sql_query(
            query,
            connection
        )

        print(result.to_string(index=False))

        output_blocks.append(
            "\n".join(
                [
                    "=" * 70,
                    name,
                    "=" * 70,
                    query.strip(),
                    "",
                    result.to_string(index=False),
                ]
            )
        )

    OUTPUT_FILE.write_text(
        "\n\n".join(output_blocks),
        encoding="utf-8"
    )

    print(
        f"\nQuery outputs saved to: {OUTPUT_FILE}"
    )

    return queries


# ============================================================
# PANDAS READ_SQL + MERGE
# ============================================================

def demonstrate_pandas_operations(connection):
    """
    Demonstrate:
        1. pd.read_sql()
        2. pd.merge()
        3. Equivalent join results
    """

    print("\n")
    print("=" * 70)
    print("PANDAS VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Read two query results using pd.read_sql
    # --------------------------------------------------------

    expensive_query = """
        SELECT
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp > 20
        ORDER BY price_gbp DESC;
    """

    expensive_df = pd.read_sql(
        expensive_query,
        connection
    )

    print("\nDataFrame 1 - pd.read_sql:")
    print(expensive_df.head(10).to_string(index=False))

    category_query = """
        SELECT
            category_id,
            category_name
        FROM categories
        ORDER BY category_name;
    """

    categories_df = pd.read_sql(
        category_query,
        connection
    )

    print("\nDataFrame 2 - pd.read_sql:")
    print(categories_df.to_string(index=False))

    # --------------------------------------------------------
    # Load base tables into pandas
    # --------------------------------------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    cats_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    # --------------------------------------------------------
    # Reproduce JOIN using pd.merge()
    # --------------------------------------------------------

    pandas_join = pd.merge(
        books_df,
        cats_df,
        on="category_id",
        how="inner"
    )

    pandas_join = pandas_join[
        [
            "title",
            "category_name",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
        ]
    ]

    pandas_join = pandas_join.sort_values(
        by=[
            "category_name",
            "rating",
            "price_gbp",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).head(10).reset_index(drop=True)

    # --------------------------------------------------------
    # SQL JOIN result
    # --------------------------------------------------------

    sql_join = pd.read_sql(
        """
        SELECT
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.price_gbp DESC
        LIMIT 10;
        """,
        connection
    )

    sql_join = sql_join.reset_index(
        drop=True
    )

    print("\nSQL JOIN result:")
    print(sql_join.to_string(index=False))

    print("\nPandas pd.merge() result:")
    print(pandas_join.to_string(index=False))

    # --------------------------------------------------------
    # Normalize types before comparison
    # --------------------------------------------------------

    sql_compare = sql_join.copy()
    pandas_compare = pandas_join.copy()

    sql_compare["in_stock"] = (
        sql_compare["in_stock"]
        .astype(bool)
    )

    pandas_compare["in_stock"] = (
        pandas_compare["in_stock"]
        .astype(bool)
    )

    sql_compare["price_gbp"] = (
        sql_compare["price_gbp"]
        .astype(float)
        .round(2)
    )

    pandas_compare["price_gbp"] = (
        pandas_compare["price_gbp"]
        .astype(float)
        .round(2)
    )

    sql_compare["price_inr"] = (
        sql_compare["price_inr"]
        .astype(float)
        .round(2)
    )

    pandas_compare["price_inr"] = (
        pandas_compare["price_inr"]
        .astype(float)
        .round(2)
    )

    equivalent = sql_compare.equals(
        pandas_compare
    )

    print(
        f"\nDo SQL JOIN and pd.merge() match? "
        f"{equivalent}"
    )

    if not equivalent:
        print(
            "\nNote: The two DataFrames should contain "
            "the same rows and values. Check sorting/types "
            "if this prints False."
        )

    return sql_join, pandas_join


# ============================================================
# DATABASE VALIDATION
# ============================================================

def validate_database(connection):
    """
    Perform basic database validation.
    """

    cursor = connection.cursor()

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print("\n========== DATABASE VALIDATION ==========")

    print(
        f"Number of books: {book_count}"
    )

    print(
        f"Number of categories: {category_count}"
    )

    print(
        f"Required minimum books met: "
        f"{book_count >= 60}"
    )

    print(
        f"Required minimum categories met: "
        f"{category_count >= 3}"
    )

    # Check for invalid ratings.
    invalid_ratings = cursor.execute(
        """
        SELECT COUNT(*)
        FROM books
        WHERE rating NOT BETWEEN 1 AND 5
        """
    ).fetchone()[0]

    print(
        f"Invalid ratings: {invalid_ratings}"
    )

    # Check for invalid prices.
    invalid_prices = cursor.execute(
        """
        SELECT COUNT(*)
        FROM books
        WHERE price_gbp IS NULL
           OR price_gbp <= 0
        """
    ).fetchone()[0]

    print(
        f"Invalid prices: {invalid_prices}"
    )

    # Check INR conversion.
    conversion_errors = cursor.execute(
        """
        SELECT COUNT(*)
        FROM books
        WHERE ABS(price_inr - price_gbp * ?) > 0.01
        """,
        (FIXED_GBP_TO_INR,)
    ).fetchone()[0]

    print(
        f"Incorrect INR conversions: "
        f"{conversion_errors}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ZEPTO DATA PIPELINE")
    print("=" * 70)

    print(
        f"\nFixed conversion rate: "
        f"1 GBP = {FIXED_GBP_TO_INR:.2f} INR"
    )

    # --------------------------------------------------------
    # 1. Scrape
    # --------------------------------------------------------

    raw_df = scrape_books()

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    cleaned_df = clean_data(
        raw_df
    )

    # --------------------------------------------------------
    # 3. Database
    # --------------------------------------------------------

    connection = create_database(
        cleaned_df
    )

    try:

        # ----------------------------------------------------
        # 4. SQL queries
        # ----------------------------------------------------

        run_sql_queries(
            connection
        )

        # ----------------------------------------------------
        # 5. Pandas operations
        # ----------------------------------------------------

        demonstrate_pandas_operations(
            connection
        )

        # ----------------------------------------------------
        # 6. Validation
        # ----------------------------------------------------

        validate_database(
            connection
        )

    finally:
        connection.close()

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nDatabase: {DB_PATH}"
    )

    print(
        f"SQL output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

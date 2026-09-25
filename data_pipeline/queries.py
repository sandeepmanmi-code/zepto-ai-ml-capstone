from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path(__file__).resolve().parent / "output" / "books.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "query_outputs.txt"
"""
SQL queries used by the Zepto Data Pipeline.

These queries demonstrate:
1. SELECT + WHERE
2. ORDER BY + LIMIT
3. DISTINCT
4. IN
5. BETWEEN
6. JOIN
"""

QUERY_1 = """
SELECT title, price_gbp, rating, in_stock
FROM books
WHERE price_gbp > 20
ORDER BY price_gbp DESC;
"""


QUERY_2 = """
SELECT title, price_gbp, rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""


QUERY_3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""


QUERY_4 = """
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
"""


QUERY_5 = """
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
"""


QUERY_6 = """
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

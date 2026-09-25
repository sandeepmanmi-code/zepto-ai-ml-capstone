Data Pipeline — Zepto
Overview

This module implements an end-to-end data-engineering pipeline:

scrape → clean → transform → normalize → store → query → analyse

The source is books.toscrape.com, a public website designed for scraping practice. The pipeline automatically discovers the available book categories and scrapes the first three categories, following pagination within each category.

No login, API key, paid service, or manual copy-pasting is required.

Requirements

Python 3.9+ is recommended.

Install the module dependencies with:

pip install -r data_pipeline/requirements.txt


The module uses:

requests for HTTP requests

BeautifulSoup from beautifulsoup4 for HTML parsing

pandas for cleaning and tabular analysis

Python's built-in sqlite3 for the relational database

Run the pipeline

From the repository root:

python data_pipeline/scrape_and_load.py


The script:

Discovers book categories.

Selects three categories.

Scrapes all paginated listing pages for those categories.

Extracts title, price, rating, availability, and category.

Cleans the extracted fields.

Converts GBP prices to INR.

Creates a normalized SQLite database.

Inserts the category and book records.

Saves the database to:

data_pipeline/output/books.db


The pipeline validates that at least 60 valid book rows and at least three categories are produced.

Fixed currency conversion

The project-defined conversion rate is:

1 GBP = 105.50 INR


This is an artificial fixed baseline specified by the assignment. It is not retrieved from a currency API.

Therefore:

price_inr = price_gbp × 105.50


The calculation is performed after parsing price_gbp as a numeric value and is rounded to two decimal places.

Cleaning decisions
Price

The source price contains a currency symbol such as £51.77.

The symbol and any surrounding text are removed using a regular expression and the result is converted to float.

If a numeric price cannot be parsed, the pipeline uses median imputation, as required by the assignment.

Rating

The source provides ratings as text:

One
Two
Three
Four
Five


These are mapped to integers:

One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5


Unexpected numeric rating values are median-imputed.

Availability

Availability text containing In stock is converted to:

True


Availability text containing Out of stock is converted to:

False


If availability cannot be interpreted, the row is dropped. A boolean availability value has no meaningful numeric median, so dropping an unparseable availability record is preferable to inventing its stock status.

Data types

The resulting dataframe contains:

title: string

price_gbp: float

price_inr: float

rating: integer from 1–5

in_stock: boolean

category: string

Database design

The SQLite database is normalized into two related tables.

categories
Column	Type	Constraint
category_id	INTEGER	Primary key
category_name	TEXT	UNIQUE, NOT NULL
books
Column	Type	Constraint
book_id	INTEGER	Primary key
title	TEXT	NOT NULL
price_gbp	REAL	NOT NULL
price_inr	REAL	NOT NULL
rating	INTEGER	1–5
in_stock	INTEGER	0/1
category_id	INTEGER	Foreign key

The relationship is:

categories.category_id
        │
        │ 1-to-many
        ▼
books.category_id


Category names are therefore stored once in categories, rather than being repeatedly stored for every book.

Foreign-key enforcement is enabled with:

PRAGMA foreign_keys = ON;

SQL queries

Run:

python data_pipeline/run_queries.py


The script executes six queries.

Query 1 — SELECT and WHERE

Retrieves highly rated books:

SELECT title, price_gbp, rating, in_stock
FROM books
WHERE rating >= 4
ORDER BY rating DESC, price_gbp DESC
LIMIT 10;


This demonstrates SELECT, WHERE, ORDER BY, and LIMIT.

Query 2 — ORDER BY and LIMIT

Retrieves the ten most expensive books:

SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;

Query 3 — DISTINCT

Retrieves the distinct ratings present in the database:

SELECT DISTINCT rating
FROM books
ORDER BY rating;

Query 4 — BETWEEN

Retrieves books whose GBP price falls within a specified range:

SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 10 AND 30
ORDER BY price_gbp;

Query 5 — IN

Retrieves highly rated books using an IN condition:

SELECT title, rating, category_id
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, title
LIMIT 15;

Query 6 — JOIN

Joins books with their categories:

SELECT
    c.category_name,
    b.title,
    b.price_gbp,
    b.price_inr,
    b.rating,
    b.in_stock
FROM books AS b
JOIN categories AS c
    ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC, b.title
LIMIT 20;


The complete query text and generated output are written to:

data_pipeline/output/query_outputs.txt

pandas SQL and merge validation

The query runner demonstrates two ways of obtaining the same joined result.

First, the SQL JOIN is loaded directly with:

pd.read_sql(...)


The two source tables are then loaded independently into pandas and joined using:

books_df.merge(
    categories_df,
    on="category_id",
    how="inner"
)


The resulting dataframes are sorted identically and compared with:

sql_join_comparable.equals(pandas_join_df)


The script prints:

JOIN RESULTS EQUIVALENT: True


when the SQL and pandas implementations produce equivalent results.

Reproducibility

The SQLite database can be regenerated from scratch at any time by running:

python data_pipeline/pipeline.py


The existing database is removed and recreated, so the pipeline does not depend on a manually prepared database file.

The query outputs can then be regenerated with:

python data_pipeline/queries.py

Design summary

The module deliberately separates extraction, transformation, storage, and analysis responsibilities.

The scraper deals only with retrieving and parsing source HTML. The cleaning layer converts semi-structured strings into typed analytical fields. The database layer stores categories and books using a normalized primary-key/foreign-key design. Finally, the query runner demonstrates relational SQL analysis and verifies that an equivalent join can be reproduced using pandas.

The fixed 1 GBP = 105.50 INR conversion is implemented locally so the required result is deterministic and does not depend on an external API or live exchange rate.
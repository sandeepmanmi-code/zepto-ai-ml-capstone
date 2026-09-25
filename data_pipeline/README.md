Module 1 — Data Pipeline
Overview

This module implements an end-to-end data engineering pipeline for Zepto-style catalog analysis.

The pipeline performs the following stages:

books.toscrape.com
        |
        v
Web Scraping
        |
        v
Raw DataFrame
        |
        v
Data Cleaning
        |
        v
GBP -> INR Conversion
        |
        v
Normalized SQLite Database
        |
        +--------------------+
        |                    |
        v                    v
       SQL               pandas
     Queries             Analysis
        |                    |
        +---------+----------+
                  |
                  v
          Validation Output


The implementation uses:

requests for HTTP requests

BeautifulSoup for HTML parsing

pandas for data cleaning and analysis

sqlite3 for the relational database

SQL for querying

pd.read_sql() and pd.merge() for pandas-based validation

Project Structure
data_pipeline/
├── pipeline.py
├── README.md
├── requirements.txt
├── queries.py
├── books.db
└── outputs/
    └── query_outputs.txt

Data Source

The data is scraped from:

https://books.toscrape.com/

Books to Scrape is a public website specifically intended for scraping practice. No login, API key, or paid service is required.

The pipeline scrapes three categories:

Travel

Mystery

Historical Fiction

The complete category pagination is followed automatically.

The resulting dataset contains at least 60 books, satisfying the project requirement.

Installation

From the project root, create and activate a virtual environment.

Windows PowerShell
python -m venv .venv

.\.venv\Scripts\Activate.ps1


Install the dependencies:

pip install -r data_pipeline\requirements.txt


Alternatively, after changing into the module directory:

cd data_pipeline
pip install -r requirements.txt

Running the Pipeline

From the data_pipeline directory:

python pipeline.py


The script performs the complete process automatically:

Scrapes the book categories.

Creates a raw pandas DataFrame.

Cleans the scraped values.

Converts GBP prices to INR.

Creates the SQLite database.

Creates the normalized tables.

Inserts the cleaned records.

Executes the SQL queries.

Saves SQL results.

Reproduces the SQL JOIN using pd.merge().

Compares the SQL and pandas results.

Performs database validation.

Data Cleaning
Price

The original price is provided in GBP.

For example:

£51.77


is converted into:

51.77


and stored in:

price_gbp


The parser uses regular expression-based numeric extraction so that unexpected surrounding text does not cause the pipeline to crash.

If a price cannot be parsed, the value is treated as missing and replaced using the median of the successfully parsed prices.

This follows the assignment requirement for numeric parsing failures.

Star Rating

The website stores ratings as textual CSS classes such as:

One
Two
Three
Four
Five


These are converted into integers:

One   -> 1
Two   -> 2
Three -> 3
Four  -> 4
Five  -> 5


If an unexpected rating cannot be parsed, it is replaced with the median valid rating.

The final column is stored as an integer.

Availability

The availability text is converted into a boolean:

In stock -> True
Anything else -> False


The final column is:

in_stock


and is stored as a boolean in pandas and as 0/1 in SQLite.

Currency Conversion

The project requires the following fixed conversion rate:

1 GBP = 105.50 INR


This is an artificial project-defined baseline and is not retrieved from an external currency API.

The conversion is:

price_inr = price_gbp * 105.50


For example:

£10.00 * 105.50 = ₹1,055.00


The pipeline rounds the resulting INR value to two decimal places.

Database Design

SQLite is used as the relational database.

The database contains two normalized tables.

categories
categories
-----------------------------
category_id     PRIMARY KEY
category_name   UNIQUE

books
books
-----------------------------
book_id         PRIMARY KEY
title
price_gbp
price_inr
rating
in_stock
category_id     FOREIGN KEY


The relationship is:

categories
    |
    | category_id
    |
    +--------< books.category_id


This avoids storing the category name repeatedly in every book record and demonstrates a normalized relational design.

Foreign-key enforcement is enabled using:

PRAGMA foreign_keys = ON;

SQL Queries

Six queries are included.

Query 1 — SELECT and WHERE
SELECT title, price_gbp, rating, in_stock
FROM books
WHERE price_gbp > 20
ORDER BY price_gbp DESC;


This filters books whose GBP price is greater than 20.

Query 2 — ORDER BY and LIMIT
SELECT title, price_gbp, rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


This returns the ten most expensive books.

Query 3 — DISTINCT
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;


This demonstrates retrieval of unique categories.

Query 4 — IN
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


This demonstrates the IN operator and a subquery.

Query 5 — BETWEEN
SELECT
    title,
    price_gbp,
    rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;


This selects books with prices between £20 and £40.

Query 6 — JOIN
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


This joins the normalized books and categories tables using the foreign key.

Query Output

The executed query strings and their results are automatically written to:

outputs/query_outputs.txt


The output file is generated when:

python pipeline.py


is executed.

pandas SQL Analysis

The pipeline uses:

pd.read_sql()


to read SQL query results into pandas DataFrames.

At least two SQL results are loaded independently into pandas.

For example:

expensive_df = pd.read_sql(
    expensive_query,
    connection
)


and:

categories_df = pd.read_sql(
    category_query,
    connection
)

SQL JOIN vs pandas.merge()

The SQL JOIN is independently reproduced using pandas.

The two base tables are loaded into pandas:

books_df = pd.read_sql(...)
categories_df = pd.read_sql(...)


They are then joined using:

pandas_join = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


The resulting DataFrame is sorted using the same ordering as the SQL query.

The pipeline then compares the SQL JOIN result and the pd.merge() result:

Do SQL JOIN and pd.merge() match? True


This demonstrates that the relational join can be reproduced using pandas without issuing the JOIN query.

Data Validation

The pipeline performs the following checks:

Total number of books is at least 60.

At least three categories are present.

Ratings are between 1 and 5.

Prices are non-null and positive.

INR conversion follows the required fixed rate.

SQL JOIN and pandas merge() produce equivalent results.

Design Decisions
Scraping

requests and BeautifulSoup were selected because they are lightweight and directly satisfy the project requirements.

Pagination is handled programmatically so no manual copy-pasting is required.

Cleaning

Numeric parsing failures are handled with median imputation because the assignment specifically requires numeric fields to use median imputation when parsing fails.

Rows with missing title or category are dropped because these are essential descriptive fields and cannot be meaningfully imputed.

Database

SQLite was selected because it is lightweight, requires no server configuration, and is included with Python.

The database is normalized into separate category and book tables connected through a foreign key.

Currency

The required project-defined conversion rate is used:

1 GBP = 105.50 INR


No external currency API is required.

SQL and pandas

SQL is used for relational querying, while pandas is used to demonstrate that the same relational operation can be reproduced in memory using pd.merge().

Expected Output

A successful run should finish with:

PIPELINE COMPLETED SUCCESSFULLY


and produce:

books.db


and:

outputs/query_outputs.txt


The database can be inspected using any SQLite-compatible database viewer.

Reproducibility

The SQLite database can be regenerated from scratch by running:

python pipeline.py


The pipeline removes the previous books.db, scrapes the source again, cleans the data, recreates the schema, and reloads the database.

Therefore, the database does not depend on manual data entry.
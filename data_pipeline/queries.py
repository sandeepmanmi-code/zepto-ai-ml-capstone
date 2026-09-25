from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path(__file__).resolve().parent / "output" / "books.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "query_outputs.txt"


QUERIES = {
    "Q1 - SELECT and WHERE": """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC, price_gbp DESC
        LIMIT 10;
    """,

    "Q2 - ORDER BY and LIMIT": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    "Q3 - DISTINCT": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,

    "Q4 - BETWEEN": """
        SELECT title, price_gbp, price_inr
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp;
    """,

    "Q5 - IN": """
        SELECT title, rating, category_id
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        LIMIT 15;
    """,

    "Q6 - JOIN": """
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
    """,
}


def run_queries():
    connection = sqlite3.connect(DB_PATH)

    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
            for name, query in QUERIES.items():
                print("=" * 80)
                print(name)
                print("=" * 80)
                print(query.strip())

                output.write("=" * 80 + "\n")
                output.write(name + "\n")
                output.write("=" * 80 + "\n")
                output.write(query.strip() + "\n\n")

                result = pd.read_sql(query, connection)

                print(result.to_string(index=False))
                print()

                output.write(result.to_string(index=False))
                output.write("\n\n")

            # ----------------------------------------------------------------
            # pd.read_sql demonstration
            # ----------------------------------------------------------------
            join_query = QUERIES["Q6 - JOIN"]

            sql_join_df = pd.read_sql(
                join_query,
                connection,
            )

            # Read source tables independently.
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
                connection,
            )

            categories_df = pd.read_sql(
                """
                SELECT category_id, category_name
                FROM categories
                """,
                connection,
            )

            # Reproduce the JOIN entirely in pandas.
            pandas_join_df = (
                books_df
                .merge(
                    categories_df,
                    on="category_id",
                    how="inner",
                )
                [
                    [
                        "category_name",
                        "title",
                        "price_gbp",
                        "price_inr",
                        "rating",
                        "in_stock",
                    ]
                ]
                .sort_values(
                    ["category_name", "rating", "title"],
                    ascending=[True, False, True],
                )
                .head(20)
                .reset_index(drop=True)
            )

            sql_join_comparable = (
                sql_join_df
                .copy()
                .sort_values(
                    ["category_name", "rating", "title"],
                    ascending=[True, False, True],
                )
                .reset_index(drop=True)
            )

            equivalent = sql_join_comparable.equals(
                pandas_join_df
            )

            print("=" * 80)
            print("pd.read_sql JOIN RESULT")
            print("=" * 80)
            print(sql_join_comparable.to_string(index=False))

            print("\n" + "=" * 80)
            print("pd.merge JOIN RESULT")
            print("=" * 80)
            print(pandas_join_df.to_string(index=False))

            print("\n" + "=" * 80)
            print(f"JOIN RESULTS EQUIVALENT: {equivalent}")
            print("=" * 80)

            output.write("=" * 80 + "\n")
            output.write("pd.read_sql JOIN RESULT\n")
            output.write("=" * 80 + "\n")
            output.write(sql_join_comparable.to_string(index=False))
            output.write("\n\n")

            output.write("=" * 80 + "\n")
            output.write("pd.merge JOIN RESULT\n")
            output.write("=" * 80 + "\n")
            output.write(pandas_join_df.to_string(index=False))
            output.write("\n\n")

            output.write(
                f"JOIN RESULTS EQUIVALENT: {equivalent}\n"
            )

            if not equivalent:
                raise AssertionError(
                    "SQL JOIN and pandas merge results do not match."
                )

    finally:
        connection.close()


if __name__ == "__main__":
    run_queries()

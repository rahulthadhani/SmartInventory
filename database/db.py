import sqlite3
import os

DB_PATH = "smartinventory.db"


def get_connection():
    """
    Creates and returns a connection to the SQLite database.
    SQLite stores the entire database as a single file on disk.
    The file is created automatically if it doesn't exist yet.
    """
    conn = sqlite3.connect(DB_PATH)

    # This makes rows behave like dictionaries so you can
    # access columns by name e.g. row["brand"] instead of row[2]
    conn.row_factory = sqlite3.Row

    return conn


def initialize_database():
    """
    Creates the products table if it doesn't already exist.
    This is safe to call every time the app starts — it won't
    overwrite or delete existing data.

    Table columns:
      id           - auto-incrementing primary key
      barcode      - the scanned UPC/EAN value (must be unique)
      brand        - product brand name
      product_name - full product name
      product_type - category or type e.g. "Energy Drink"
      size         - size or variant e.g. "12 fl oz"
      ocr_text     - raw text extracted by OCR (Week 3)
      description  - LLM-generated description (Week 3)
      timestamp    - when the record was created
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode      TEXT UNIQUE NOT NULL,
            brand        TEXT,
            product_name TEXT,
            product_type TEXT,
            size         TEXT,
            ocr_text     TEXT,
            description  TEXT,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    conn.close()
    print(f"Database initialized at {os.path.abspath(DB_PATH)}")

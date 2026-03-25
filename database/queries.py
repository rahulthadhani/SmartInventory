from database.db import get_connection


def find_product_by_barcode(barcode_value):
    """
    Looks up a product in the database by its barcode value.

    Returns the matching row as a dict if found, or None if
    the product doesn't exist in the database yet.

    This is the core check that decides whether to show an
    existing record or trigger the OCR pipeline.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products WHERE barcode = ?", (barcode_value,))
    row = cursor.fetchone()
    conn.close()

    if row:
        # Convert the Row object to a plain dict for easy use
        return dict(row)

    return None


def insert_product(product_data):
    """
    Inserts a new product record into the database.

    product_data should be a dict with keys matching the
    column names: barcode, brand, product_name, product_type,
    size, ocr_text, description.

    Returns the new record's id if successful, None if it fails.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO products
                (barcode, brand, product_name, product_type, size, ocr_text, description)
            VALUES
                (:barcode, :brand, :product_name, :product_type, :size, :ocr_text, :description)
        """,
            product_data,
        )

        conn.commit()
        new_id = cursor.lastrowid
        print(f"Product saved to database with id {new_id}")
        return new_id

    except Exception as e:
        print(f"Error inserting product: {e}")
        conn.rollback()
        return None

    finally:
        conn.close()


def update_product(barcode_value, updated_fields):
    """
    Updates an existing product record by barcode.

    updated_fields is a dict containing only the columns
    you want to change e.g. {"brand": "Celsius", "size": "12 fl oz"}

    Dynamically builds the SET clause so you don't have to
    pass every column — only the ones being updated.
    """
    if not updated_fields:
        print("No fields to update.")
        return

    conn = get_connection()
    cursor = conn.cursor()

    # Build "brand = ?, product_name = ?" etc. from the dict keys
    set_clause = ", ".join([f"{key} = ?" for key in updated_fields.keys()])
    values = list(updated_fields.values()) + [barcode_value]

    try:
        cursor.execute(f"UPDATE products SET {set_clause} WHERE barcode = ?", values)
        conn.commit()
        print(f"Product {barcode_value} updated successfully.")

    except Exception as e:
        print(f"Error updating product: {e}")
        conn.rollback()

    finally:
        conn.close()


def get_all_products():
    """
    Returns every product in the database as a list of dicts.
    Used for displaying the full inventory.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def delete_product(barcode_value):
    """
    Deletes a product record by barcode.
    Useful for removing test entries during development.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM products WHERE barcode = ?", (barcode_value,))
    conn.commit()
    conn.close()
    print(f"Product {barcode_value} deleted.")

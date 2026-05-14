from database.supabase_client import supabase


def initialize_database():
    """
    Verifies the Supabase connection is working.
    The Product table already exists in your Supabase project.
    """
    try:
        supabase.table("Product").select("id").limit(1).execute()
        print("Connected to Supabase database.")
    except Exception as e:
        print(f"Supabase connection error: {e}")

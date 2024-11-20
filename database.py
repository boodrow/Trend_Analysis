# database.py

import sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd
from config import DATABASE, NUM_DAYS_BACK
from logger import logger
import urllib.parse
import time


def get_connection(max_retries=5, initial_delay=2):
    """
    Establishes a connection to the PostgreSQL database with retries.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            # URL-encode the password to handle special characters
            password = urllib.parse.quote_plus(DATABASE['PASSWORD'])
            connection_url = f"postgresql+psycopg2://{DATABASE['USER']}:{password}@{DATABASE['HOST']}:{DATABASE['PORT']}/{DATABASE['NAME']}"
            engine = create_engine(connection_url)

            # Test connection using an executable object
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                fetched = result.scalar()
                if fetched != 1:
                    raise ValueError("Database connection test failed.")

            logger.info("Database connection established.")
            return engine
        except Exception as e:
            attempt += 1
            logger.error(f"Attempt {attempt} - Error connecting to the database: {e}")
            time.sleep(initial_delay * (2 ** (attempt - 1)))  # Exponential backoff

    logger.critical("All attempts to connect to the database have failed.")
    raise ConnectionError("Failed to connect to the database after multiple attempts.")


def initialize_last_processed_table(engine):
    """
    Initializes the 'last_processed' table if it does not exist.
    """
    try:
        # Use engine.begin() to start a transaction
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS last_processed (
                    frequency VARCHAR(10) PRIMARY KEY,
                    last_timestamp TIMESTAMP WITH TIME ZONE
                );
            """))
        logger.info("Ensured 'last_processed' table exists.")
    except Exception as e:
        logger.error(f"Error initializing 'last_processed' table: {e}")


def get_last_processed_timestamp(engine, frequency):
    """
    Retrieves the last processed timestamp for a given frequency.
    """
    try:
        query = text("SELECT last_timestamp FROM last_processed WHERE frequency = :frequency")
        with engine.connect() as connection:
            result = connection.execute(query, {'frequency': frequency}).fetchone()
            if result:
                return result[0]  # Access by index instead of key
            else:
                return None
    except Exception as e:
        logger.error(f"Error fetching last processed timestamp for {frequency}: {e}")
        return None


def update_last_processed_timestamp(engine, frequency, timestamp):
    """
    Updates the last processed timestamp for a given frequency.
    """
    try:
        query = text("""
            INSERT INTO last_processed (frequency, last_timestamp)
            VALUES (:frequency, :last_timestamp)
            ON CONFLICT (frequency) DO UPDATE SET last_timestamp = :last_timestamp
        """)
        # Use engine.begin() to start a transaction
        with engine.begin() as connection:
            connection.execute(query, {'frequency': frequency, 'last_timestamp': timestamp})
        logger.info(f"Persisted last processed timestamp for {frequency}: {timestamp}")
    except Exception as e:
        logger.error(f"Error updating last processed timestamp for {frequency}: {e}")


def fetch_data(engine, table_name, last_processed_timestamp=None, limit=None):
    """
    Fetch records from the table based on the last processed timestamp.
    If last_processed_timestamp is None, fetch records from the last NUM_DAYS_BACK days.
    """
    try:
        if last_processed_timestamp:
            query = text(f"SELECT * FROM {table_name} WHERE timestamp > :last_timestamp ORDER BY timestamp ASC")
            params = {'last_timestamp': last_processed_timestamp}
        else:
            # Fetch records within the last NUM_DAYS_BACK days if no timestamp is set
            num_days = NUM_DAYS_BACK
            query = text(
                f"SELECT * FROM {table_name} WHERE timestamp >= NOW() - INTERVAL '{num_days} days' ORDER BY timestamp ASC")
            params = {}

        if limit:
            query = text(f"{query.text} LIMIT :limit")
            params['limit'] = limit

        df = pd.read_sql(query, engine, params=params)
        logger.info(f"Fetched {len(df)} new records from {table_name}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()


def ensure_columns(engine, table_name, columns):
    """
    Ensure that specified columns exist in the database table.
    """
    try:
        with engine.begin() as conn:
            # Get existing columns in the table
            result = conn.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND LOWER(table_name) = LOWER(:table_name)
                """), {'table_name': table_name}
            )
            existing_columns = [row['column_name'] for row in result.mappings()]

            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    conn.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
                    )
                    logger.info(f"Added '{column_name}' column to {table_name}.")
                else:
                    logger.info(f"'{column_name}' column already exists in {table_name}.")
    except Exception as e:
        logger.error(f"Error ensuring columns in table {table_name}: {e}")


def update_trends(engine, table_name, df):
    """
    Bulk update the 'trend' and 'predicted_close' columns in the database based on the DataFrame.
    """
    try:
        if not {'trend', 'predicted_close', 'timestamp'}.issubset(df.columns):
            missing_cols = {'trend', 'predicted_close', 'timestamp'}.difference(df.columns)
            logger.warning(f"Missing columns {missing_cols} in dataframe for {table_name}.")
            return
        df_to_update = df[['timestamp', 'trend', 'predicted_close']]
        if df_to_update.empty:
            logger.info(f"No 'trend' or 'predicted_close' data to update for {table_name}.")
            return
        # Convert 'timestamp' to string if necessary
        if df_to_update['timestamp'].dtype == 'datetime64[ns]':
            df_to_update['timestamp'] = df_to_update['timestamp'].astype(str)
        with engine.begin() as conn:
            # Create a temporary table
            conn.execute(text("DROP TABLE IF EXISTS temp_trends"))
            conn.execute(text("CREATE TEMPORARY TABLE temp_trends (timestamp TEXT, trend TEXT, predicted_close REAL)"))
            # Insert data into temporary table
            df_to_update.to_sql('temp_trends', conn, if_exists='append', index=False)
            # Update the original table using the temporary table
            conn.execute(text(f"""
                UPDATE {table_name}
                SET trend = temp_trends.trend,
                    predicted_close = temp_trends.predicted_close
                FROM temp_trends
                WHERE {table_name}.timestamp::text = temp_trends.timestamp
            """))
            # Drop the temporary table
            conn.execute(text("DROP TABLE IF EXISTS temp_trends"))
        logger.info(f"Bulk updated 'trend' and 'predicted_close' columns in {table_name}.")
    except Exception as e:
        logger.error(f"Error updating trends in table {table_name}: {e}")
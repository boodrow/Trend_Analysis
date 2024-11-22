# database.py

import sqlalchemy
from sqlalchemy import create_engine, text
import pandas as pd
from config import DATABASE, NUM_DAYS_BACK, TIMESTAMP_DIR, FLOAT_TOLERANCE, BATCH_UPDATE
from logger import logger
import urllib.parse
import time
import os
import pickle


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


def fetch_data(engine, table_name, columns='*', start_time=None, end_time=None, limit=None):
    """
    Fetch records from the table based on the specified time range and columns.
    If columns='*', fetch all columns.
    """
    try:
        if isinstance(columns, list):
            columns = ', '.join(columns)
        elif isinstance(columns, str) and columns != '*':
            columns = columns
        else:
            columns = '*'

        if start_time and end_time:
            query = text(f"""
                SELECT {columns}
                FROM {table_name}
                WHERE timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp ASC
            """)
            params = {'start_time': start_time, 'end_time': end_time}
        elif start_time:
            query = text(f"""
                SELECT {columns}
                FROM {table_name}
                WHERE timestamp >= :start_time
                ORDER BY timestamp ASC
            """)
            params = {'start_time': start_time}
        else:
            # Fetch records within the last NUM_DAYS_back days if no timestamp is set
            num_days = NUM_DAYS_BACK
            query = text(
                f"SELECT {columns} FROM {table_name} WHERE timestamp >= NOW() - INTERVAL '{num_days} days' ORDER BY timestamp ASC"
            )
            params = {}

        if limit:
            query = text(f"{query.text} LIMIT :limit")
            params['limit'] = limit

        df = pd.read_sql(query, engine, params=params, parse_dates=['timestamp'])
        logger.info(f"Fetched {len(df)} records from {table_name}.")
        return df
    except Exception as e:
        logger.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()


def ensure_columns(engine, table_name, columns):
    """
    Ensure that specified columns exist in the database table.
    """
    try:
        with engine.connect() as conn:
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


def update_trends(engine, table_name, df, freq):
    """
    Update the 'trend', 'predicted_close', 'sma', and 'ema' columns in the database based on the DataFrame.
    Updates are performed in batches to ensure efficiency.
    """
    try:
        required_columns = {'trend', 'predicted_close', 'sma', 'ema', 'timestamp'}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            logger.warning(f"Missing columns {missing_cols} in dataframe for {table_name}.")
            return

        df_to_update = df[['timestamp', 'trend', 'predicted_close', 'sma', 'ema']].copy()
        if df_to_update.empty:
            logger.info(f"No 'trend', 'predicted_close', 'sma', or 'ema' data to update for {table_name}.")
            return

        # Ensure 'timestamp' is timezone-aware and in UTC
        if df_to_update['timestamp'].dt.tz is None:
            df_to_update['timestamp'] = df_to_update['timestamp'].dt.tz_localize('UTC')
        else:
            df_to_update['timestamp'] = df_to_update['timestamp'].dt.tz_convert('UTC')

        # Sort by timestamp to ensure consistent order
        df_to_update.sort_values('timestamp', inplace=True)

        # Ensure 'timestamp's are unique
        if df_to_update['timestamp'].duplicated().any():
            logger.warning(f"Duplicate 'timestamp's found in DataFrame for {table_name}. Removing duplicates.")
            df_to_update = df_to_update.drop_duplicates(subset=['timestamp'])

        total_records = len(df_to_update)
        batches = [df_to_update.iloc[i:i + BATCH_UPDATE] for i in range(0, total_records, BATCH_UPDATE)]
        updated_records = 0

        with engine.connect() as conn:
            trans = conn.begin()
            try:
                for batch_num, batch in enumerate(batches, start=1):
                    update_query = text(f"""
                        UPDATE {table_name}
                        SET trend = :trend,
                            predicted_close = :predicted_close,
                            sma = :sma,
                            ema = :ema
                        WHERE timestamp = :timestamp
                    """)

                    update_data = batch.to_dict(orient='records')
                    result = conn.execute(update_query, update_data)
                    updated_records += result.rowcount
                    logger.debug(f"Batch {batch_num}: Updated {result.rowcount} records.")

                # Verification step using count method inside the same transaction
                try:
                    # Define the time range based on the DataFrame
                    start_time = df_to_update['timestamp'].min()
                    end_time = df_to_update['timestamp'].max()

                    verification_query = text(f"""
                        SELECT COUNT(*) as count
                        FROM {table_name}
                        WHERE timestamp BETWEEN :start_time AND :end_time
                          AND trend IS NOT NULL
                          AND predicted_close IS NOT NULL
                          AND sma IS NOT NULL
                          AND ema IS NOT NULL
                    """)

                    result = conn.execute(verification_query,
                                          {'start_time': start_time, 'end_time': end_time}).fetchone()

                    # Access the count based on the result type
                    if isinstance(result, tuple):
                        non_null_count = result[0] if result else 0
                    elif hasattr(result, 'count'):
                        non_null_count = result.count if result else 0
                    elif isinstance(result, dict):
                        non_null_count = result['count'] if result else 0
                    else:
                        logger.error(f"Unexpected result type: {type(result)}")
                        non_null_count = 0

                    if non_null_count < updated_records:
                        logger.error(
                            f"Verification failed: Expected at least {updated_records} non-NULL records, found {non_null_count}.")
                        raise ValueError(
                            f"Verification failed: Expected at least {updated_records} non-NULL records, found {non_null_count}.")
                    elif non_null_count > updated_records:
                        logger.warning(
                            f"Verification: Expected {updated_records} non-NULL records, but found {non_null_count}. Some records may have been previously updated.")
                        # Decide whether to raise an error or not based on acceptable discrepancy
                        # For now, we'll log a warning and not raise an error
                    else:
                        logger.info(
                            f"Verification successful: {non_null_count} non-NULL records updated in {table_name}.")

                except Exception as e:
                    logger.error(f"Error during verification for table {table_name}: {e}")
                    raise

                trans.commit()
                logger.info(
                    f"Successfully updated 'trend', 'predicted_close', 'sma', and 'ema' in {table_name} for {updated_records} records.")

            except Exception as e:
                trans.rollback()
                logger.error(f"Error during batch updates in table {table_name}: {e}")
                raise

    except Exception as e:
        logger.error(f"Error updating trends in table {table_name}: {e}")
        raise  # Re-raise the exception to stop the program as per requirement
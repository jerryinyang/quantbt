import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv, dotenv_values
import pandas as pd
import sys
import time



# Configure the connection
load_dotenv()
config = dotenv_values(".env")

db_config = {
    "host": config.get("SQL_HOST"),
    "user": config.get("SQL_USERNAME"),
    "password": config.get("SQL_PASSWORD"),
    "database": config.get("SQL_DATABASE"),
}



class DBHandler:
    def __init__(self):
        try: 
            self.connection = mysql.connector.connect(**db_config)
            self.cursor = self.connection.cursor()

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Error: Access denied.")
                raise err
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Error: Database does not exist.")
            else:
                print(f"Error: {err}")


    def create_ohlc_table(self):
        try:
            # Create the OHLC Table
            self.cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS `OHLC` (
                    `timestamp` timestamp NOT NULL,
                    `resolution` varchar(5) NOT NULL,
                    `ticker_id` varchar(10) NOT NULL,
                    `market_type` varchar(10) NOT NULL,
                    `open` float NOT NULL,
                    `high` float NOT NULL,
                    `low` float NOT NULL,
                    `close` float NOT NULL,
                    `volume` int NOT NULL,
                    PRIMARY KEY (`timestamp`,`resolution`,`ticker_id`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Contains OHLCV Data.'
                """
            )
            self.connection.commit()
            return True
        except Exception as e:
            raise e
            

    def insert_data(self, table_name: str, data: pd.DataFrame):
        # Get the table column names and order
        table_columns = self.__fetch_table_column_order(table_name)

        # TODO: Check Data is in the right format

        # Reorder the data columns
        data = data[table_columns]

        # Specify chunk size for batch processing
        chunk_size = 10000
        start_time = time.time()

        # Iterate over DataFrame in chunks and insert into MySQL
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            records = chunk.to_records(index=False)
            columns = ', '.join(map(str, table_columns))
            values = ', '.join(map(str, records))

            insert_query = f"INSERT IGNORE INTO {table_name} ({columns}) VALUES {values}"

            try:
                self.cursor.execute(insert_query)
                self.connection.commit()

                # Update load bar
                progress = (i + chunk_size) / len(data) * 100
                bar_length = 50
                filled_length = int(bar_length * progress // 100)
                bar = "=" * filled_length + "-" * (bar_length - filled_length)

                # Calculate time duration and estimate remaining time
                elapsed_time = time.time() - start_time
                remaining_time = ((elapsed_time / progress) * (100 - progress)) if progress > 0 else 0

                # Convert remaining time to minutes and seconds
                minutes, seconds = divmod(remaining_time, 60)

                sys.stdout.write("\rProgress: [{}] {:.2f}% | Remaining Time: {:02}:{:02}".format(bar, progress, int(minutes), int(seconds)))
                sys.stdout.flush()

            except mysql.connector.Error as err:
                print(f"Error: {err}")
                self.connection.rollback()

        # Print a newline to complete the progress bar
        print("\nInsertion complete.")  


# # Example: Loading bar with 10 iterations
# loading_bar(10)     


    def fetch_data(self, table_name : str):
        # SQL code to fetch data from the table
        self.cursor.execute(f"SELECT * FROM {table_name}")
        return self.cursor.fetchall()


    def close_connection(self):
        # Close the database connection
        self.connection.close()


    def __fetch_table_column_order(self, table_name:str):
        assert self.connection.is_connected(), ValueError('SQL Database is not connected.')
        
        try:
            query =    f'''
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table_name}'
                ORDER BY ORDINAL_POSITION;
            '''
            self.cursor.execute(query)
            
            return [column[0] for column in self.cursor.fetchall()]

        except mysql.connector.Error as err:
            raise f"Error retreiving table columns: {err}"
        
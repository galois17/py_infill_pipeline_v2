import sqlite3
from sqlite3 import Error
import pandas as pd
import copy
from infill_pipeline.infill_error import InfillError
import infill_pipeline.utility as utility
import infill_pipeline.config as config

class DataStore:
    """ DataStore 
    """
    states = {'open': 'o', 'active': 'a', 'completed': 'c'}

    def __init__(self):
        self.name = "DataStore"

    def get_version(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return sqlite3.version
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def print_version(self, db_file):
        print(self.get_version(db_file))

    def create_schema(self, db_file):
        """ Creates the schema """
        job_schema = """
        CREATE TABLE job_queue (
                                        id INTEGER NOT NULL PRIMARY KEY,                                        
                                        name text,
                                        creation_date text,
                                        begin_date text,
                                        end_date text,
                                        state text,
                                        dat_infill_id INTEGER
                                    );
        """

        dat_schema = """
            CREATE TABLE IF NOT EXISTS "dat" (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "infill_id" INTEGER,
            "X1" REAL,
            "X2" REAL,
            "X3" REAL,
            "X4" REAL,
            "X5" REAL,
            "X6" REAL,
            "X7" REAL,
            "X8" REAL,
            "X9" REAL,
            "X10" REAL,
            "X11" REAL,
            "X12" REAL,
            "X13" REAL,
            "X14" REAL,
            "X15" REAL,
            "X16" REAL,
            "X17" REAL,
            "X18" REAL,
            "X19" REAL,
            "X20" REAL,
            "infill_type" TEXT,
            "Y1" TEXT,
            "Y2" TEXT,
            "Y3" TEXT,
            "Y4" TEXT,
            "Y5" TEXT,
            "Y6" TEXT,
            "Y7" TEXT,
            "Y8" TEXT,
            "Y9" TEXT,
            "Y10" TEXT,
            "Y11" TEXT,
            "Y12" TEXT,
            "Y13" TEXT,
            "Y14" TEXT,
            "Y15" TEXT,
            "Y16" TEXT,
            "Y17" TEXT,
            "Y18" TEXT,
            "Y19" TEXT
            );
        """
        conn = sqlite3.connect(db_file)            
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            cur.execute(job_schema)
            cur.execute(dat_schema)
            conn.commit()
        except Error as e:
            conn.rollback()
            print(e)
            raise e
        finally:
            if conn:
                conn.close()

    def insert_new_job(self, db_file, infill_id):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            open = DataStore.states['open']
            cur_time = utility.get_datetime()
            cur.execute(
                f"INSERT INTO job_queue (state, dat_infill_id, creation_date) values ('{open}', {infill_id}, {cur_time});")
            conn.commit()
        except Error as e:
            print(e)
            raise e
        finally:
            if conn:
                conn.close()

    def insert_new_infilling_job(self, db_file, infill_id):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            name = "infill"
            # Delete existing item in queue
            q = f"DELETE FROM job_queue WHERE name='{name}';"
            cur.execute(q)

            open = DataStore.states['open']
            cur_time = utility.get_datetime()
            
            q = f"INSERT INTO job_queue (state, dat_infill_id, name, creation_date) values ('{open}', {infill_id}, '{name}', {cur_time});"
            print(f"DB: {q}")
            cur.execute(q)

            conn.commit()
        except Error as e:
            conn.rollback()
            print(e)
            raise e
        finally:
            if conn:
                conn.close()

    def get_all_responses_for_design(self, db_file):
        "select y1,y2,y3,y4,y5,y6,y7,y8,y9 from dat where infill_type='design' order by infill_id;"

    def count_all_jobs(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            value = cur.execute(
                f"SELECT COUNT(*) as cnt FROM job_queue;"
            ).fetchone()
            
            return value[0]
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def get_infilling_jobs(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            open = DataStore.states['open']
            name = 'infill'
            cur.execute(
                f"SELECT dat_infill_id FROM job_queue WHERE name='{name}';"
            )
            values = cur.fetchall()            
            return values
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def count_open_infilling_jobs(self, db_file):
        conn = None
        print(f"count from db file {db_file}")
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            open = DataStore.states['open']
            name = 'infill'
            value = cur.execute(
                f"SELECT COUNT(*) as cnt FROM job_queue WHERE state='{open}' and name='{name}';"
            ).fetchone()
            
            return value[0]
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def count_open_jobs(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            open = DataStore.states['open']
            value = cur.execute(
                f"SELECT COUNT(*) as cnt FROM job_queue WHERE state='{open}';"
            ).fetchone()
            
            return value[0]
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()
    
    def get_open_jobs(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            open = DataStore.states['open']
            cur.execute(
                f"SELECT dat_infill_id FROM job_queue WHERE state='{open}';"
            )
            values = cur.fetchall()
            return values
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def update_dat(self, db_file, dict):
        dict_new = copy.deepcopy(dict)
        num_of_x_var = 20
        num_of_y_var = 19
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()

            for j in range(1, num_of_x_var+1):
                key = f"X{j}"

                if key not in dict.keys():
                    dict_new[key] = 'NULL'

            for j in range(1, num_of_y_var+1):
                key = f"Y{j}"

                if key not in dict.keys():
                    dict_new[key] = "\'\'"
            
            stmt = f"""UPDATE dat SET X1={dict_new['X1']},X2={dict_new['X2']},X3={dict_new['X3']},X4={dict_new['X4']},X5={dict_new['X5']},X6={dict_new['X6']},X7={dict_new['X7']},X8={dict_new['X8']},X9={dict_new['X9']},X10={dict_new['X10']},X11={dict_new['X11']},X12={dict_new['X12']},X13={dict_new['X13']},X14={dict_new['X14']},X15={dict_new['X15']},X16={dict_new['X16']},X17={dict_new['X17']},X18={dict_new['X18']},X19={dict_new['X19']},X20={dict_new['X20']},
                Y1={dict_new['Y1']},Y2={dict_new['Y2']},Y3={dict_new['Y3']},Y4={dict_new['Y4']},Y5={dict_new['Y5']},Y6={dict_new['Y6']},Y7={dict_new['Y7']},Y8={dict_new['Y8']},Y9={dict_new['Y9']},Y10={dict_new['Y10']},Y11={dict_new['Y11']},Y12={dict_new['Y12']},Y13={dict_new['Y13']},Y14={dict_new['Y14']},Y15={dict_new['Y15']},Y16={dict_new['Y16']},Y17={dict_new['Y17']},Y18={dict_new['Y18']},Y19={dict_new['Y19']}  WHERE infill_id={dict_new['infill_id']};"""

            cur.execute(stmt)
            conn.commit()
        except Error as e:
            print("Exception: %s " % (e))
            raise InfillError("Unable to insert row.")
        finally:
            if conn:
                conn.close()

    def update_job(self, db_file, infill_id, state):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()

            cur.execute(
                f"UPDATE job_queue SET state='{state}' WHERE dat_infill_id={infill_id};")
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def update_job_to_active(self, db_file, infill_id):
        """ Update the job to active (or 'o')"""
        self.update_job(db_file, infill_id, DataStore.states['active'])

    def update_job_to_completed(self, db_file, infill_id):
        self.update_job(db_file, infill_id, DataStore.states['completed'])

    def check_job_state(self, db_file, infill_id, should_have_state):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute(
                f"SELECT state FROM job_queue where infill_id={infill_id};")
            cur_result = cur.fetchone()
            if cur_result == should_have_state:
                return True
            else:
                return False
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def check_dat_exists(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()

            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dat';")
            if cur.rowcount > 0:
                cur_result = cur.fetchone()
                return True
            else:
                return False
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def get_new_infill_id(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            # delete all rows from table
            cur.execute("SELECT MAX(infill_id)+1 FROM dat;")
            return cur.fetchone()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def drop_all(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            # Delete all rows from table
            cur.execute('DROP TABLE dat;')
            cur.execute('DROP TABLE job_queue;')
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def delete_all(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            # Delete all rows from table
            cur.execute('DELETE FROM dat')
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def count(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) from dat')
            cur_result = cur.fetchone()
            rows = cur_result[0]
            return rows
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def get_design_points(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)

            dat_df = pd.read_sql_query("SELECT * FROM dat", conn)
            design_df = dat_df[dat_df.infill_type == config.INFILL_TYPE_DESIGN]

            return design_df
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def get_all_infill(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)

            dat_df = pd.read_sql_query(
                "SELECT * FROM dat WHERE infill_type='infill'", conn)
            #row_df = dat_df[dat_df.infill_id == infill_id]
            return dat_df
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def get_row_by_infill_id(self, db_file, infill_id):
        conn = None
        try:
            conn = sqlite3.connect(db_file)

            dat_df = pd.read_sql_query("SELECT * FROM dat", conn)
            row_df = dat_df[dat_df.infill_id == infill_id]

            return row_df
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def insert_design_init(self, db_file, design_init_df):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            design_init_df.to_sql(
                'dat', conn, schema='dat', if_exists='append', index=True, index_label='id')
            #design_init_df.to_sql('dat', conn, schema='dat', if_exists='fail', index=True, index_label='id')
        except Error as e:
            print("Not able to insert design init to table! Error is: ")
            print(e)
        finally:
            if conn:
                conn.close()

    def insert_row(self, db_file, design_with_responses_df):
        """ Insert row into 'dat'
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            design_with_responses_df.to_sql(
                'dat', conn, if_exists='append', index=False, index_label='id')
        except Error as e:
            print("Not able to insert design_with_responses_df to table! Error is: ")
            print(e)
        finally:
            if conn:
                conn.close()

    def insert_row_raw(self, db_file, insert_query, tuple):
        """ Insert in row of data to datastore. x"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            sql = insert_query
            cur.execute(sql, tuple)

            conn.commit()
        except Error as e:
            print("Exception: %s " % (e))
            raise InfillError("Unable to insert row.")
        finally:
            if conn:
                conn.close()

    def delete_from_dat_by_infill_id(self, db_file, infill_id):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            # delete all rows from table
            cur.execute(f"DELETE FROM dat WHERE infill_id={infill_id};")
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()  

    def delete_all_from_job_queue(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            # delete all rows from table
            cur.execute('DELETE FROM job_queue')
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def __str__(self):
        return f'DataStore is named {self.name}'

    def __repr__(self):
        return f'DataStore(name={self.name})'

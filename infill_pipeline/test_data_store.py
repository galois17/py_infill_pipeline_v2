import unittest
import yaml
import os
import sqlite3
from sqlite3 import Error
import pandas as pd
import numpy as np

from infill_pipeline.infill_pipeline import InfillPipeline
import infill_pipeline.config as config
from infill_pipeline.infill_error import InfillError
from infill_pipeline.data_store import DataStore

class TestDataStore(unittest.TestCase):

    def setup(self):
        with open("unittests_data/test_config.yaml", 'r') as stream:
            config.data_loaded = yaml.safe_load(stream)

    def clean_db(self, db_file):
        conn = None
        sql_file = None
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            cursor.execute("drop table dat;")
            cursor.execute("drop table job_queue;")
            conn.commit()
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()
            if sql_file:
                sql_file.close()

        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            sql_file = open("unittests_data/dump.sql")
            sql_as_string = sql_file.read()
            cursor.executescript(sql_as_string)

            print("Clean db!!!!")
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()
            if sql_file:
                sql_file.close()

    def test_instance(self):
        self.setup()

        run_folder = config.data_loaded['system']['run_folder']
        sqlite_db = config.data_loaded['system']['sqlite_db']

        ds = DataStore()
        ver = ds.get_version(os.path.join(run_folder, sqlite_db))
        print(ver)
        self.assertTrue(len(ver) > 0)

    # @unittest.skip("")
    def test_get_all_design_points(self):
        self.setup()

        run_folder = config.data_loaded['system']['run_folder']
        sqlite_db = config.data_loaded['system']['sqlite_db']

        ds = DataStore()
        self.clean_db(os.path.join(run_folder, sqlite_db))
        design_df = ds.get_design_points(os.path.join(run_folder, sqlite_db))
        print(design_df)

        self.assertTrue(design_df.size > 0)

    def test_insert_raw_row(self):
        self.setup()

        run_folder = config.data_loaded['system']['run_folder']
        sqlite_db = config.data_loaded['system']['sqlite_db']

        ds = DataStore()
        self.clean_db(os.path.join(run_folder, sqlite_db))

        insert_query = ''' INSERT INTO dat(infill_id, X1,X2,X3,X4)
              VALUES(?,?,?,?,?) '''

        row_count = ds.count(os.path.join(run_folder, sqlite_db))
        count = row_count
        ds.insert_row_raw(os.path.join(run_folder, sqlite_db),
                          insert_query,
                          (count + 1, 1.1, 2.2, 3.3, 4.4))
        new_row_count = ds.count(os.path.join(run_folder, sqlite_db))
        print("Old count=%d, New count=%d" % (row_count, new_row_count))

        self.assertEqual(new_row_count, row_count + 1)

        row_df = ds.get_row_by_infill_id(
            os.path.join(run_folder, sqlite_db), count+1)

        # Convert pandas dataframe to just a single value
        self.assertTrue(row_df['infill_id'].values[0] == count+1)

        with self.assertRaises(InfillError) as context:
            ds.insert_row_raw(os.path.join(run_folder, sqlite_db),
                              insert_query,
                              (count + 1, 1.1, 2.2, 3.3, 4.4))

    def test_insert(self):
        self.setup()

        run_folder = config.data_loaded['system']['run_folder']
        sqlite_db = config.data_loaded['system']['sqlite_db']

        ds = DataStore()
        self.clean_db(os.path.join(run_folder, sqlite_db))

        # Create pandas dataframe
        dat = {'X1': [1], 'X2': [2], 'X3': [3], 'X4': [4], 'X5': [5], 'X6': [6], 'X7': [7], 'X8': [8],
               'X9': [9], 'X10': [10], 'X11': [11], 'X12': [12], 'X13': [13], 'X14': [14], 'X15': [15],
               'Y1': [None]
               }
        dat_df = pd.DataFrame(dat)

        c1 = ds.count(os.path.join(run_folder, sqlite_db))
        ds.insert_row(os.path.join(run_folder, sqlite_db), dat_df)
        c2 = ds.count(os.path.join(run_folder, sqlite_db))

        self.assertTrue(c2 == (c1 + 1), f"c1={c1} and c2={c2}\n")

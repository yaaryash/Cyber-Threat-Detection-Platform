import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from threatsentry.exception.exception import ThreatDetectionException
from threatsentry.logger.logger import logger
from dotenv import load_dotenv

load_dotenv()
ca=certifi.where()
MONGO_DB_URL=os.getenv("MONGO_DB_URI")

class PhishingDataExtract():
    """
    Handles extraction of phishing data from CSV
    and loading it into MongoDB Atlas.
    """
    def __init__(self):
        try:
            logger.info("PhishingDataExtract initialized")
        except Exception as e:
            raise ThreatDetectionException(e, sys)
    
    def csv_to_json_convertor(self, file_path: str) -> list:
        """
        Reads phishing CSV dataset and converts
        each row to a JSON record for MongoDB insertion.
        """
        try:
            logger.info(f"Reading CSV from: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            logger.info(f"Converted {len(records)} records to JSON")
            return records
        except Exception as e:
            raise ThreatDetectionException(e, sys)
        
    
    def insert_data_mongodb(self, records: list,
                             database: str,
                             collection: str) -> int:
        """
        Inserts JSON records into specified
        MongoDB database and collection.
        """
        try:
            self.database=database
            self.collection=collection
            self.records=records
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)

if __name__== '__main__':
    FILE_PATH = "data\phisingData.csv"
    DATABASE = "CyberThreatDB"
    COLLECTION = "PhishingData"

    extractor = PhishingDataExtract()
    records = extractor.csv_to_json_convertor(file_path=FILE_PATH)
    count = extractor.insert_data_mongodb(records, DATABASE, COLLECTION)
    print(f"Total records inserted: {count}")
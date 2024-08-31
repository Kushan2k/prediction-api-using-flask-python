import pymongo
import os 
import dotenv 
dotenv.load_dotenv()

class DatabaseConnection():
    _instance = None
    def __new__(cls):
        if not cls._instance:
            client = pymongo.MongoClient(os.getenv('CONNECTION_STRING'))
            cls._instance = client
        return cls._instance
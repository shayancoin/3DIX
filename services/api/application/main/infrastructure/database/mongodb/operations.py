from abc import ABC
from typing import Dict

import motor.motor_asyncio

from application.main.config import settings
from application.main.infrastructure.database.db_interface import DataBaseOperations
from application.main.utility.config_loader import ConfigReaderInstance


class Mongodb(DataBaseOperations, ABC):

    def __init__(self):
        super(Mongodb, self).__init__()
        self.db_config = ConfigReaderInstance.yaml.read_config_from_file(
            settings.DB + '_config.yaml')
        self.connection_uri = 'mongodb://' + \
            str(self.db_config['test']['host']) + ':' + str(self.db_config['test']['port'])
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_uri)
        self.db = self.client[self.db_config.get('database', '3dix_db')]

    async def insert_single_db_record(self, record: Dict, collection_name: str = None):
        coll_name = collection_name or self.db_config.get('collection', 'default_collection')
        collection = self.db[coll_name]
        return await collection.insert_one(record)

    async def find(self, collection_name: str, query: Dict):
        collection = self.db[collection_name]
        cursor = collection.find(query)
        return await cursor.to_list(length=None)

    async def find_one(self, collection_name: str, query: Dict):
        collection = self.db[collection_name]
        return await collection.find_one(query)

    async def update_one(self, collection_name: str, query: Dict, update: Dict):
        collection = self.db[collection_name]
        return await collection.update_one(query, update)

    async def delete_one(self, collection_name: str, query: Dict):
        collection = self.db[collection_name]
        return await collection.delete_one(query)

    # Legacy methods to satisfy interface if needed, or remove if interface allows
    async def fetch_single_db_record(self, unique_id: str):
        pass

    async def update_single_db_record(self, record: Dict):
        pass

    async def update_multiple_db_record(self, record: Dict):
        pass

    async def fetch_multiple_db_record(self, unique_id: str):
        pass

    async def insert_multiple_db_record(self, record: Dict):
        pass

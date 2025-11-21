from abc import ABC
from typing import Dict

import motor.motor_asyncio

from application.main.config import settings
from application.main.infrastructure.database.db_interface import DataBaseOperations
from application.main.utility.config_loader import ConfigReaderInstance


class Mongodb(DataBaseOperations, ABC):

    def __init__(self):
        """
        Initialize the Mongodb instance by loading configuration, constructing a MongoDB URI, creating an AsyncIOMotorClient, and selecting the target database.
        
        Loads configuration from a YAML file named "<DB>_config.yaml" (where <DB> is settings.DB), builds the connection URI using the configured host and port under the 'test' section, creates an AsyncIOMotorClient with that URI, and sets the `db` attribute to the configured database (defaults to "3dix_db" if not present).
        
        Attributes:
            db_config (dict): Parsed YAML configuration for the database.
            connection_uri (str): MongoDB connection URI constructed from config.
            client (AsyncIOMotorClient): Motor async client connected to the MongoDB server.
            db (Database): Selected Motor database instance.
        """
        super(Mongodb, self).__init__()
        self.db_config = ConfigReaderInstance.yaml.read_config_from_file(
            settings.DB + '_config.yaml')
        self.connection_uri = 'mongodb://' + \
            str(self.db_config['test']['host']) + ':' + str(self.db_config['test']['port'])
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_uri)
        self.db = self.client[self.db_config.get('database', '3dix_db')]

    async def insert_single_db_record(self, record: Dict, collection_name: str = None):
        """
        Insert a single document into the configured MongoDB collection.
        
        Parameters:
            record (Dict): Document to insert.
            collection_name (str, optional): Target collection name; if omitted, the configured default collection is used.
        
        Returns:
            The insertion result object; its `inserted_id` attribute contains the new document's id.
        """
        coll_name = collection_name or self.db_config.get('collection', 'default_collection')
        collection = self.db[coll_name]
        return await collection.insert_one(record)

    async def find(self, collection_name: str, query: Dict):
        """
        Finds documents in the specified MongoDB collection that match the given filter.
        
        Parameters:
        	collection_name (str): Name of the target MongoDB collection.
        	query (Dict): MongoDB filter document used to match documents.
        
        Returns:
        	documents (List[Dict]): List of documents from the collection that match `query`.
        """
        collection = self.db[collection_name]
        cursor = collection.find(query)
        return await cursor.to_list(length=None)

    async def find_one(self, collection_name: str, query: Dict):
        """
        Retrieve a single document from the specified collection that matches the given MongoDB filter.
        
        Parameters:
            collection_name (str): Name of the target MongoDB collection.
            query (Dict): MongoDB filter document used to match the desired document.
        
        Returns:
            dict or None: The matched document as a mapping if found, `None` if no document matches.
        """
        collection = self.db[collection_name]
        return await collection.find_one(query)

    async def update_one(self, collection_name: str, query: Dict, update: Dict):
        """
        Update a single document in the specified collection that matches the provided query.
        
        Parameters:
            collection_name (str): Name of the collection to operate on.
            query (Dict): Filter that identifies the document to update.
            update (Dict): Update document (e.g., using MongoDB update operators) describing the modifications to apply.
        
        Returns:
            update_result: The result of the update operation; contains attributes such as `matched_count` and `modified_count`.
        """
        collection = self.db[collection_name]
        return await collection.update_one(query, update)

    async def delete_one(self, collection_name: str, query: Dict):
        """
        Delete a single document matching the given query from the specified collection.
        
        Returns:
            DeleteResult: Result of the delete operation; `deleted_count` is the number of documents deleted.
        """
        collection = self.db[collection_name]
        return await collection.delete_one(query)

    # Legacy methods to satisfy interface if needed, or remove if interface allows
    async def fetch_single_db_record(self, unique_id: str):
        pass

    async def update_single_db_record(self, record: Dict):
        pass

    async def update_multiple_db_record(self, record: Dict):
        """
        Retains the legacy signature for updating multiple database records; currently a no-op placeholder.
        
        Parameters:
            record (Dict): Data describing the intended multi-record update (e.g., filter and update payload), preserved for interface compatibility.
        """
        pass

    async def fetch_multiple_db_record(self, unique_id: str):
        """
        Legacy placeholder for fetching multiple database records by unique identifier.
        
        This async method is intentionally unimplemented and preserved for interface compatibility.
        
        Parameters:
            unique_id (str): Identifier used to locate related records across collections or documents.
        """
        pass

    async def insert_multiple_db_record(self, record: Dict):
        pass
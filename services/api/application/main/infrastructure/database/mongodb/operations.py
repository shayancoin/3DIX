from abc import ABC
from typing import Dict

import motor.motor_asyncio

from application.main.config import settings
from application.main.infrastructure.database.db_interface import DataBaseOperations
from application.main.utility.config_loader import ConfigReaderInstance


class Mongodb(DataBaseOperations, ABC):

    def __init__(self):
        """
        Set up the MongoDB client and select the target database using the configured settings.
        
        Initializes instance attributes required for database operations based on the YAML configuration named "<DB>_config.yaml" (where <DB> is settings.DB).
        
        Attributes:
            db_config (dict): Parsed YAML configuration for the database.
            connection_uri (str): MongoDB connection URI constructed from the configuration.
            client (AsyncIOMotorClient): Motor async client connected to the MongoDB server.
            db (Database): Selected Motor database instance (defaults to "3dix_db" if not specified in config).
        """
        super(Mongodb, self).__init__()
        self.db_config = ConfigReaderInstance.yaml.read_config_from_file(
            settings.DB + '_config.yaml')
        # Struct returned by config loader; access via attributes
        host = getattr(self.db_config.test, "host", "localhost")
        port = getattr(self.db_config.test, "port", 27017)
        database_name = getattr(self.db_config, "database", "3dix_db")
        self.connection_uri = f"mongodb://{host}:{port}"
        self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_uri)
        self.db = self.client[database_name]

    async def insert_single_db_record(self, record: Dict, collection_name: str = None):
        """
        Inserts a single document into the configured MongoDB collection.
        
        Parameters:
            record (Dict): Document to insert.
            collection_name (str, optional): Target collection name; if omitted, uses the configured default collection.
        
        Returns:
            insertion_result: The insertion result object; its `inserted_id` attribute contains the new document's id.
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
        Delete a single document that matches the provided MongoDB filter from the given collection.
        
        Parameters:
            collection_name (str): Name of the target collection.
            query (Dict): MongoDB filter document used to match the document to delete.
        
        Returns:
            DeleteResult: Result of the delete operation; `deleted_count` is the number of documents deleted.
        """
        collection = self.db[collection_name]
        return await collection.delete_one(query)

    # Legacy methods to satisfy interface if needed, or remove if interface allows
    async def fetch_single_db_record(self, unique_id: str):
        pass

    async def update_single_db_record(self, record: Dict):
        """
        Placeholder for the legacy single-record update API that intentionally performs no operation.
        
        Kept for backward compatibility; accepts the legacy `record` parameter but does not modify the database or return a value.
        
        Parameters:
            record (Dict): Legacy record payload accepted for compatibility; this parameter is ignored.
        """
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
        """
        Compatibility placeholder that preserves the legacy signature for inserting multiple records.
        
        This method is a no-op maintained for backward compatibility; the provided `record` argument is accepted for signature compatibility and is not used.
        Parameters:
            record (Dict): Data representing one or more documents; accepted for compatibility but ignored.
        """
        pass

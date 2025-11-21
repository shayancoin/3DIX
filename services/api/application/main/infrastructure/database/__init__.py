from application.main.infrastructure.database.mongodb.operations import Mongodb
from application.main.infrastructure.database.postgresql.operations import PostgreSQL

DataBaseToUse = {
    'mongodb': Mongodb(),
    'postgresql': PostgreSQL(),
    'postgres': PostgreSQL(),  # Alias for postgresql
}

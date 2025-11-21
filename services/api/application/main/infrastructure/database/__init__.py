from application.main.infrastructure.database.mongodb.operations import Mongodb
from application.main.infrastructure.database.postgresql.operations import Postgresql

DataBaseToUse = {
    'mongodb': Mongodb(),
    'postgresql': Postgresql(),
}

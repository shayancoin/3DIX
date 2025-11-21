class LoggerInstance(object):
    def __new__(cls):
        from application.main.utility.logger.custom_logging import LogHandler
        return LogHandler()


class IncludeAPIRouter(object):
    def __new__(cls):
        """
        Create and return a FastAPI APIRouter configured with the application's sub-routers.
        
        Includes and registers the health check, hello world, response manager, question classification,
        image classification, vibe, projects, jobs, internal jobs, and custom objects routers under their
        respective URL prefixes and tags (e.g., '/api/v1', '/api/v1/vibe', '/api/v1/projects', '/api/v1/jobs',
        '/api/v1/internal/jobs', '/api/v1/custom-objects').
        
        Returns:
            APIRouter: An APIRouter instance with all listed sub-routers included and tagged.
        """
        from application.main.routers.health_checks import router as router_health_check
        from application.main.routers.hello_world import router as router_hello_world
        from application.main.routers.api_response import router as response_manager_test
        from application.main.routers.question_classifier import router as router_question_classification
        from application.main.routers.image_classifier import router as router_image_classification
        from application.main.routers import vibe, projects, rooms
        from application.main.routers.jobs import router as router_jobs
        from application.main.routers.jobs_internal import router as router_jobs_internal
        from application.main.routers.custom_objects import router as router_custom_objects
        from fastapi.routing import APIRouter
        router = APIRouter()
        router.include_router(router_health_check, prefix='/api/v1', tags=['health_check'])
        router.include_router(router_hello_world, prefix='/api/v1', tags=['hello_world'])
        router.include_router(response_manager_test, prefix='/api/v1', tags=['response_manager'])
        router.include_router(router_question_classification, prefix='/api/v1', tags=['question_classification'])
        router.include_router(router_image_classification, prefix='/api/v1', tags=['image_classification'])
        router.include_router(router_vibe, prefix='/api/v1/vibe', tags=['vibe'])
        router.include_router(router_projects, prefix='/api/v1/projects', tags=['projects'])
        router.include_router(router_jobs, prefix='/api/v1/jobs', tags=['jobs'])
        router.include_router(router_jobs_internal, prefix='/api/v1/internal/jobs', tags=['jobs-internal'])
        router.include_router(router_custom_objects, prefix='/api/v1/custom-objects', tags=['custom-objects'])
        return router


class DataBaseInstance(object):
    def __new__(cls):
        from application.main.infrastructure.database import db
        return db.DataBase()


# instance creation
logger_instance = LoggerInstance()
db_instance = DataBaseInstance()
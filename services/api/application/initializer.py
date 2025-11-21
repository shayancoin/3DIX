class LoggerInstance(object):
    def __new__(cls):
        from application.main.utility.logger.custom_logging import LogHandler
        return LogHandler()


class IncludeAPIRouter(object):
    def __new__(cls):
        """
        Builds and returns an APIRouter that aggregates the application's sub-routers.
        
        Includes mounted sub-routers for health checks, hello world, API response handling, question classification,
        vibe functionality, project and room stubs, and custom objects under the API prefixes used by the application.
        
        Returns:
            APIRouter: The assembled router containing the included sub-routers (health_check, hello_world,
            response_manager, question_classification, vibe, projects-stub, rooms-stub, custom-objects).
        """
        from application.main.routers.health_checks import router as router_health_check
        from application.main.routers.hello_world import router as router_hello_world
        from application.main.routers.api_response import router as response_manager_test
        from application.main.routers.question_classifier import router as router_question_classification
        # from application.main.routers.image_classifier import router as router_image_classification
        from application.main.routers import vibe
        # Stub routers to satisfy frontend contract without real DB
        from application.main.routers.stub import (
            router as router_stub_projects,
            rooms_router as router_stub_rooms,
        )
        # Optional: keep jobs/custom_objects wiring if needed later
        # from application.main.routers.jobs import router as router_jobs
        # from application.main.routers.jobs_internal import router as router_jobs_internal
        from application.main.routers.custom_objects import router as router_custom_objects
        from fastapi.routing import APIRouter
        router = APIRouter()
        router.include_router(router_health_check, prefix='/api/v1', tags=['health_check'])
        router.include_router(router_hello_world, prefix='/api/v1', tags=['hello_world'])
        router.include_router(response_manager_test, prefix='/api/v1', tags=['response_manager'])
        router.include_router(router_question_classification, prefix='/api/v1', tags=['question_classification'])
        # router.include_router(router_image_classification, prefix='/api/v1', tags=['image_classification'])
        router.include_router(vibe.router, prefix='/api/v1/vibe', tags=['vibe'])
        router.include_router(router_stub_projects, prefix='/api/v1/projects', tags=['projects-stub'])
        router.include_router(router_stub_rooms, prefix='/api/v1', tags=['rooms-stub'])
        # router.include_router(router_jobs, prefix='/api/v1/jobs', tags=['jobs'])
        # router.include_router(router_jobs_internal, prefix='/api/v1/internal/jobs', tags=['jobs-internal'])
        router.include_router(router_custom_objects, prefix='/api/v1', tags=['custom-objects'])
        return router


class DataBaseInstance(object):
    def __new__(cls):
        from application.main.infrastructure.database import db
        return db.DataBase()


# instance creation
logger_instance = LoggerInstance()
db_instance = DataBaseInstance()
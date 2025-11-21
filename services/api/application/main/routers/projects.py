from typing import List
from fastapi import APIRouter, HTTPException, Request, status
from application.main.schemas.project import Project, ProjectCreate, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/", response_model=Project, status_code=status.HTTP_201_CREATED)
async def create_project(request: Request, project_in: ProjectCreate):
    project = Project(**project_in.dict())
    # Use insert_single_db_record. Note: It expects a dict and returns an InsertOneResult
    # We need to handle the _id. The Project model generates one, but mongo might want to generate it or we force it.
    # The Project model has `id` aliased to `_id`.
    project_dict = project.dict(by_alias=True)

    # We need to ensure we are using the correct collection for projects.
    # The current DBOperations uses a single collection defined in config.
    # This is a limitation. We might need to update DBOperations to support multiple collections
    # or just use the single collection for now with a "type" field?
    # The implementation plan said "Design DB schema".
    # MongoDB is schemaless, but we should probably separate collections.
    # However, DBOperations.insert_single_db_record uses `self.db_config['collection']`.
    # I should probably update DBOperations to accept a collection name or update the config to have multiple collections.
    # For now, to avoid changing DBOperations too much, I will assume we might need to override the collection
    # or just add a "doc_type" field to distinguish entities if we are forced to use one collection.
    # BUT, a better approach is to modify DBOperations to take an optional collection name.

    # Let's check operations.py again.
    # It gets collection from `self.db_config['collection']`.

    # I'll add a `doc_type` field to the project dict for now to distinguish it in the single collection
    # if I can't easily change the collection name in the call.
    # Wait, I can change operations.py to accept collection_name.

    # Let's stick to the current DBOperations for a moment.
    # If I look at `insert_single_db_record`, it uses `self.db_config['collection']`.
    # I will modify `operations.py` to accept an optional `collection_name` argument.
    # This is a better design.

    # But first, let's write the router assuming I'll fix operations.py.

    await request.app.state.db_operations.insert_single_db_record(project_dict, collection_name="projects")
    return project

@router.get("/", response_model=List[Project])
async def list_projects(request: Request):
    # fetch_multiple_db_record currently takes a unique_id? That seems wrong for "list all".
    # Let's check operations.py.
    # `fetch_multiple_db_record(self, unique_id: str)`
    # It seems the current DBOperations is very limited/specific.
    # I need to significantly improve DBOperations to support generic CRUD.

    # For now, I'll assume I will fix DBOperations to have `find` method.
    projects = await request.app.state.db_operations.find("projects", {})
    return [Project(**p) for p in projects]

@router.get("/{project_id}", response_model=Project)
async def get_project(request: Request, project_id: str):
    project_data = await request.app.state.db_operations.find_one("projects", {"_id": project_id})
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")
    return Project(**project_data)

@router.put("/{project_id}", response_model=Project)
async def update_project(request: Request, project_id: str, project_in: ProjectUpdate):
    project_data = await request.app.state.db_operations.find_one("projects", {"_id": project_id})
    if not project_data:
        raise HTTPException(status_code=404, detail="Project not found")

    update_data = project_in.dict(exclude_unset=True)
    if update_data:
        update_data["updated_at"] = datetime.utcnow()
        await request.app.state.db_operations.update_one("projects", {"_id": project_id}, {"$set": update_data})
        project_data.update(update_data)

    return Project(**project_data)

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(request: Request, project_id: str):
    result = await request.app.state.db_operations.delete_one("projects", {"_id": project_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return None

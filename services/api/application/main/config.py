# configs.py
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR_PATH = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_FILE = os.environ.get("APP_ENV_FILE")

if DEFAULT_ENV_FILE:
    DEFAULT_ENV_FILE = str(Path(DEFAULT_ENV_FILE))
else:
    candidate_files = [BASE_DIR_PATH / ".env", BASE_DIR_PATH / ".env.dev"]
    for candidate in candidate_files:
        if candidate.exists():
            DEFAULT_ENV_FILE = str(candidate)
            break
    else:
        DEFAULT_ENV_FILE = str(candidate_files[0])

BASE_SETTINGS_CONFIG = {
    "env_file": DEFAULT_ENV_FILE,
    "env_file_encoding": "utf-8",
}


class AppConfig(BaseModel):
    """Application configurations."""

    VAR_A: int = 33
    VAR_B: float = 22.0

    # question classification settings
    SPACY_MODEL_IN_USE: str = "en_core_web_sm"

    # all the directory level information defined at app config level
    # we do not want to pollute the env level config with these information
    # this can change on the basis of usage

    BASE_DIR: Path = BASE_DIR_PATH

    SETTINGS_DIR: Path = BASE_DIR.joinpath('settings')
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    LOGS_DIR: Path = BASE_DIR.joinpath('logs')
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    MODELS_DIR: Path = BASE_DIR.joinpath('models')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # local cache directory to store images or text file
    CACHE_DIR: Path = BASE_DIR.joinpath('cache')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # question classification model to use
    CLASSIFICATION_MODEL: Path = MODELS_DIR.joinpath(
        'question_classification.sav')


class GlobalConfig(BaseSettings):
    """Global configurations."""

    # These variables will be loaded from the .env file. However, if
    # there is a shell environment variable having the same name,
    # that will take precedence.

    model_config = SettingsConfigDict(**BASE_SETTINGS_CONFIG)

    APP_CONFIG: AppConfig = AppConfig()

    API_NAME: Optional[str] = Field(None, env="API_NAME")
    API_DESCRIPTION: Optional[str] = Field(None, env="API_DESCRIPTION")
    API_VERSION: Optional[str] = Field(None, env="API_VERSION")
    API_DEBUG_MODE: Optional[bool] = Field(None, env="API_DEBUG_MODE")

    # define global variables with the Field class
    ENV_STATE: str = Field("dev", env="ENV_STATE")

    # logging configuration file
    LOG_CONFIG_FILENAME: Optional[str] = Field(None, env="LOG_CONFIG_FILENAME")

    # environment specific variables do not need the Field class
    HOST: Optional[str] = None
    PORT: Optional[int] = None
    LOG_LEVEL: Optional[str] = None

    DB: Optional[str] = None

    MOBILENET_V2: Optional[str] = None
    INCEPTION_V3: Optional[str] = None


class DevConfig(GlobalConfig):
    """Development configurations."""

    model_config = SettingsConfigDict(
        **BASE_SETTINGS_CONFIG,
        env_prefix="DEV_",
    )


class ProdConfig(GlobalConfig):
    """Production configurations."""

    model_config = SettingsConfigDict(
        **BASE_SETTINGS_CONFIG,
        env_prefix="PROD_",
    )


class FactoryConfig:
    """Returns a config instance depending on the ENV_STATE variable."""

    def __init__(self, env_state: Optional[str]):
        self.env_state = env_state

    def __call__(self):
        state = (self.env_state or "dev").lower()
        if state == "prod":
            return ProdConfig()
        return DevConfig()


settings = FactoryConfig(GlobalConfig().ENV_STATE)()
# print(settings.__repr__())

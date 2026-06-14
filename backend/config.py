import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./echosense.db"
    )
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    APP_ENV: str      = os.getenv("APP_ENV", "development")
    UPLOAD_DIR: str   = os.path.join(
        os.path.dirname(__file__), "..", "data", "samples"
    )


settings = Settings()
import os

DB_PATH = "career_connector.db"
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JOBDATA_PATH = os.path.join(os.path.dirname(__file__), "Career_connector_app", "JobData.csv")
import os

DB_PATH = "Your Database Path"
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "Your Algorithm"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

JOBDATA_PATH = os.path.join(os.path.dirname(__file__), "Your Folder Name", "Your Training Data")

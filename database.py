import sqlite3
from .config import DB_PATH

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # recruiters
    c.execute('''CREATE TABLE IF NOT EXISTS recruiters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        company TEXT NOT NULL
    )''')
    # jobs
    c.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        skills TEXT NOT NULL,
        salary TEXT NOT NULL,
        location TEXT NOT NULL,
        eligibility TEXT NOT NULL,
        recruiter_email TEXT NOT NULL
    )''')
    # students
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        skills TEXT NOT NULL,
        discoverable INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

import datetime
import jwt
from passlib.hash import bcrypt
from .config import JWT_SECRET, JWT_ALGORITHM

def hash_password(password):
    return bcrypt.hash(password)

def verify_password(password, hashed):
    return bcrypt.verify(password, hashed)

def create_jwt(payload):
    payload = dict(payload)
    payload["exp"] = datetime.datetime.utcnow() + datetime.timedelta(hours=6)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

from pydantic import BaseModel
from typing import List

class Instance(BaseModel):
    token: str
    bucket_name: str
    object_name: str
    phrase: str

class GCSRequest(BaseModel):
    instances: List[Instance]

class ResponseModel(BaseModel):
    code: int
    description: str
    result: str
    score: float

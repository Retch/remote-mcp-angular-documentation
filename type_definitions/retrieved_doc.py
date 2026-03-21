from pydantic import BaseModel

class RetrievedDoc(BaseModel):
    text: str
    score: float

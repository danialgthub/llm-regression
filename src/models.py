from pydantic import BaseModel
from typing import List, Dict

class PromptConfig(BaseModel):
    version: str
    timestamp: str
    system_prompt: str
    examples: list

class EmailClassification(BaseModel):
    category: str
    summary: str

from typing import Dict
import openai
from pydantic import BaseModel

class EmailClassification(BaseModel):
    category: str
    summary: str

def classify_email(email_text: str, prompt_config: Dict) -> EmailClassification:
    # Construct the full prompt from YAML config
    system_prompt = prompt_config["system_prompt"]
    few_shot_examples = prompt_config.get("examples", [])
    
    messages = [{"role": "system", "content": system_prompt}]
    for ex in few_shot_examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": email_text})

    # --- TEMPORARY STUB RETURN ---
    # Comment out the OpenAI call for now
    return EmailClassification(
        category="general",
        summary="Stub summary for testing"
    )
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=messages,
    #     temperature=0
    # )
    
    # # Parse response into structured JSON
    # output = response["choices"][0]["message"]["content"]
    # parsed = EmailClassification.parse_raw(output)
    # return parsed

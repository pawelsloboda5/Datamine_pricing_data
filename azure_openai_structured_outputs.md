# Azure OpenAI Structured Outputs

_Last updated: 02/28/2025_  
_Contributors: 5_

Structured outputs enforce a model's response to follow a provided JSON Schema, ensuring strict adherence unlike the older JSON mode, which only guaranteed valid JSON. This feature is ideal for use cases such as function calling, structured data extraction, and multi-step workflow automation.

> **Note:** Structured outputs are **not supported** for:
> - Bring-your-own-data scenarios
> - Assistants or Azure AI Agents
> - `gpt-4o-audio-preview` and `gpt-4o-mini-audio-preview` (version: `2024-12-17`)

---

## 📌 Supported Models

| Model Name         | Version          |
|--------------------|------------------|
| `gpt-4.5-preview`  | 2025-02-27       |
| `o3-mini`          | 2025-01-31       |
| `o1`               | 2024-12-17       |
| `gpt-4o-mini`      | 2024-07-18       |
| `gpt-4o`           | 2024-08-06       |
| `gpt-4o`           | 2024-11-20       |

---

## 🧠 API Support

- Introduced in API version: `2024-08-01-preview`
- Supported in latest GA API: `2024-10-21`

---

## 🚀 Getting Started

### Install Required Libraries

```bash
pip install openai pydantic --upgrade
```

### Example (Python)

```python
from pydantic import BaseModel
from openai import AzureOpenAI
import os

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2024-10-21"
)

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="MODEL_DEPLOYMENT_NAME",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed
print(event)
print(completion.model_dump_json(indent=2))
```

**Output:**
```json
name='Science Fair' date='Friday' participants=['Alice', 'Bob']
```

...

> _Make sure to validate schemas carefully and avoid unsupported JSON Schema keywords for reliable results._
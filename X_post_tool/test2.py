from dotenv import load_dotenv
load_dotenv()

import os

print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))
print("KEY:", bool(os.getenv("LANGSMITH_API_KEY")))

from langsmith import traceable


@traceable(name="test_trace")
def test_trace():
    return "LangSmith is working"
    


print(test_trace())
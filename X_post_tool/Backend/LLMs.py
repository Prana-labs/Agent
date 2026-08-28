from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
#from groq import Groq
#from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import tempfile
import os

load_dotenv()

prompt = PromptTemplate.from_template(
    """
    Answer the question using the following context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

#model = ChatGroq(
#    model="llama-3.1-8b-instant",
#    temperature=0.2,
#    api_key=os.getenv("GROQ_API_KEY")
#)

#model = OpenAI(
#  base_url = "https://integrate.api.nvidia.com/v1",
#  api_key = "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"
#)


parser = StrOutputParser()

chain = prompt | model | parser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import tempfile
import os

load_dotenv()
# =========================================================
# LLM
# =========================================================

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

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser
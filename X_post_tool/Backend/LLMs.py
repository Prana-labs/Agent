import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

parser = StrOutputParser()

chain = prompt | model | parser
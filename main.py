import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor

from vector.vector import products_tool

load_dotenv()

SYSTEM_PROMPT = (
    "Eres un asistente experto en buscar productos de la tienda de zapatos. "
    "Ayudas a usuarios a responder preguntas sobre productos y encontrar información relevante. "
    "Usas herramientas para buscar información relevante y responder consultas de usuarios. "
    "Cuando tengas la respuesta, proporciona la información del producto y su stock si está disponible."
)

app = FastAPI()

class ProductAgentRequest(BaseModel):
    text: str
    provider: Optional[str] = "gemini"
    model: Optional[str] = None
    temperature: Optional[float] = 0.2

class ProductAgentResponse(BaseModel):
    result: str

def get_llm(provider: str, model: Optional[str], temperature: float):
    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GOOGLE_API_KEY para usar Gemini.")
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            api_key=api_key,
            temperature=temperature,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
        )
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL")
        return ChatOllama(
            model=model or "llama3",
            base_url=base_url,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")

@app.post("/products_agent_search", response_model=ProductAgentResponse)
def products_agent_endpoint(req: ProductAgentRequest):
    llm = get_llm(req.provider, req.model, req.temperature)
    tools = [products_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Ejecuta el agente de forma completamente automática
    result = executor.invoke({"input": req.text})
    # El resultado puede estar en diferentes campos según el modelo
    if isinstance(result, dict) and "output" in result:
        return ProductAgentResponse(result=str(result["output"]))
    return ProductAgentResponse(result=str(result))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

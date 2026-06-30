from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from orchestrator import TriAgentOrchestrator
from models.schemas import ProcessRequest, ProcessResult, HealthResponse
from config.settings import settings

orchestrators = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting up...")
    orchestrators["deepseek"] = TriAgentOrchestrator(
        model="deepseek-chat", provider="deepseek", rules_path=settings.rules_path)
    orchestrators["gemma4"] = TriAgentOrchestrator(
        model="gemma4-local", provider="llamacpp_b", rules_path=settings.rules_path)
    orchestrators["qwen"] = TriAgentOrchestrator(
        model="qwen-local", provider="llamacpp_b", rules_path=settings.rules_path)
    print(f"rules in store: {orchestrators['deepseek'].get_rules_count()}")
    yield
    print("shutting down")


app = FastAPI(
    title="TriAgent",
    description="Three-layer multi-agent framework for agricultural process automation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        rules_loaded=orchestrators["deepseek"].get_rules_count(),
        llm_model="deepseek-chat | gemma4-local | qwen-local"
    )


@app.post("/process", response_model=ProcessResult)
async def process_request(request: ProcessRequest, model: str = "deepseek"):
    if model not in orchestrators:
        raise HTTPException(status_code=400, detail=f"use 'deepseek', 'gemma4', or 'qwen'")
    try:
        return orchestrators[model].process(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-documents")
async def load_documents(documents_path: str = "data/fao_documents", background_tasks: BackgroundTasks = None):
    def _load():
        orchestrators["deepseek"].load_documents(documents_path)
    if background_tasks:
        background_tasks.add_task(_load)
        return {"message": "loading in background"}
    _load()
    return {"message": "done", "rules": orchestrators["deepseek"].get_rules_count()}


@app.get("/rules")
async def list_rules():
    store = orchestrators["deepseek"].rule_store
    return {
        "total": len(store),
        "rules": [
            {"rule_id": rid, "trigger": r["trigger"], "source": r["metadata"].get("source", "unknown")}
            for rid, r in store.rules.items()
        ]
    }


@app.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    store = orchestrators["deepseek"].rule_store
    if rule_id not in store.rules:
        raise HTTPException(status_code=404, detail="not found")
    return store.rules[rule_id]


@app.get("/tools")
async def list_tools():
    return {"tools": orchestrators["deepseek"].execution_agent.get_available_tools()}

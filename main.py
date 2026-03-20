"""
BigTechGroup CAITO AI Assistant
AI South Hack 2026
Architecture: Classifier → Calculator → State Machine → Persona → LLM
"""

import os
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI, APIError

from classifier import classify, Classification
from calculator import calculate_scenario, compare_scenarios, ScenarioResult
from state_machine import SessionState, update_state
from persona import build_system_prompt

load_dotenv()

# --- Config ---
API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
PORT = int(os.getenv("PORT", "8000"))
MAX_HISTORY = 20
SESSION_TTL = 3600  # 1 час


# --- Session ---
@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)
    state: SessionState = field(default_factory=SessionState)
    last_active: float = field(default_factory=time.time)


sessions: dict[str, Session] = {}


def cleanup_sessions():
    """Удаляем сессии старше SESSION_TTL."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in expired:
        del sessions[sid]


def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, Session]:
    cleanup_sessions()  # чистим при каждом запросе
    if session_id and session_id in sessions:
        sessions[session_id].last_active = time.time()
        return session_id, sessions[session_id]
    sid = session_id or str(uuid.uuid4())
    sessions[sid] = Session()
    return sid, sessions[sid]


def trim_history(messages: list[dict]) -> list[dict]:
    if len(messages) > MAX_HISTORY:
        return messages[-MAX_HISTORY:]
    return messages


# --- Pydantic ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    scenario: Optional[str] = None
    classification: Optional[str] = None


# --- App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 CAITO Assistant v2 starting on port {PORT}")
    print(f"   Model: {MODEL}")
    print(f"   Architecture: Classifier → Calculator → State Machine → Persona → LLM")
    yield
    print("👋 CAITO Assistant shutting down")


app = FastAPI(
    title="BigTechGroup CAITO AI Assistant",
    description="AI-ассистент CAITO с decision engine, калькулятором и state machine",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

# --- Static files ---
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / "index.html"))


# --- Pipeline ---

def run_pipeline(message: str, session: Session) -> tuple[str, Classification, ScenarioResult | None]:
    """
    Полный pipeline обработки сообщения:
    1. Classify → 2. Calculate → 3. Update state → 4. Build prompt
    Возвращает (system_prompt, classification, calc_result)
    """

    # 1. Classify
    classification = classify(message, pressure_count=session.state.pressure_count)

    # 2. Calculate (только если new_fact с числами)
    calc_result = None
    comparison_table = None

    if classification.category == "new_fact" and classification.extracted_params:
        params = classification.extracted_params
        calc_result = calculate_scenario(
            scenario_base=session.state.current_scenario if session.state.current_scenario in ("A", "B", "C") else "B",
            capex_cut_pct=params.get("capex_cut_pct", 0),
            order_growth_pct=params.get("order_growth_pct", 20),
            precision_override=params.get("precision"),
        )
        # Также пересчитываем все сценарии для сравнения
        comparison = compare_scenarios(
            capex_cut_pct=params.get("capex_cut_pct", 0),
            order_growth_pct=params.get("order_growth_pct", 20),
        )
        comparison_table = comparison.markdown_table

    # Для "что если" вопросов тоже считаем
    if classification.category == "question" and classification.extracted_params:
        params = classification.extracted_params
        if any(k in params for k in ["capex_cut_pct", "budget_mln", "order_growth_pct", "precision"]):
            calc_result = calculate_scenario(
                scenario_base=session.state.current_scenario if session.state.current_scenario in ("A", "B", "C") else "B",
                capex_cut_pct=params.get("capex_cut_pct", 0),
                order_growth_pct=params.get("order_growth_pct", 20),
                precision_override=params.get("precision"),
            )
            comparison = compare_scenarios(
                capex_cut_pct=params.get("capex_cut_pct", 0),
                order_growth_pct=params.get("order_growth_pct", 20),
            )
            comparison_table = comparison.markdown_table

    # 3. Update state
    session.state = update_state(session.state, classification, calc_result)

    # 4. Build dynamic prompt
    system_prompt = build_system_prompt(
        state=session.state,
        classification=classification,
        calc_result=calc_result,
        comparison_table=comparison_table,
    )

    return system_prompt, classification, calc_result


async def _generate_response(message: str, session_id: Optional[str] = None) -> tuple[str, str, str, str]:
    """Генерирует ответ. Возвращает (response, session_id, scenario, classification)."""
    sid, session = get_or_create_session(session_id)

    # Pipeline
    system_prompt, classification, calc_result = run_pipeline(message, session)

    session.messages.append({"role": "user", "content": message})
    trimmed = trim_history(session.messages)

    messages = [{"role": "system", "content": system_prompt}] + trimmed

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=messages,
            temperature=0.3,
        )
        assistant_text = response.choices[0].message.content

        session.messages.append({"role": "assistant", "content": assistant_text})
        session.messages = trim_history(session.messages)

        return assistant_text, sid, session.state.current_scenario, classification.category

    except (APIError, Exception) as e:
        if session.messages and session.messages[-1]["role"] == "user":
            session.messages.pop()
        # Fallback — отвечаем без LLM
        fallback = _get_fallback_response(classification.category, session.state)
        session.messages.append({"role": "assistant", "content": fallback})
        return fallback, sid, session.state.current_scenario, classification.category


FALLBACK_RESPONSES = {
    "pressure": "Моя позиция не изменилась, потому что не изменились данные. Покажи новые цифры — пересчитаю.",
    "new_fact": "Принял к сведению. Нужна пауза для пересчёта — повтори через минуту.",
    "question": "Коротко: моя позиция — Сценарий B (отложить на 2–3 месяца для ретрейна). Precision на полной базе = 0.312, это ниже порога 0.350. Масштабировать рано.",
}


def _get_fallback_response(category: str, state: SessionState) -> str:
    base = FALLBACK_RESPONSES.get(category, FALLBACK_RESPONSES["question"])
    return f"{base}\n\n[Текущая позиция: Сценарий {state.current_scenario}, уверенность {state.confidence*100:.0f}%]"


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "version": "2.0", "sessions_active": len(sessions)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(req.message) > 5000:
        raise HTTPException(status_code=400, detail="Message too long")
    text, sid, scenario, cls = await _generate_response(req.message.strip(), req.session_id)
    return ChatResponse(response=text, session_id=sid, scenario=scenario, classification=cls)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_v1(req: ChatRequest):
    return await chat(req)


@app.post("/chat", response_model=ChatResponse)
async def chat_short(req: ChatRequest):
    return await chat(req)


@app.get("/api/chat/stream")
async def chat_stream(message: str, session_id: Optional[str] = None):
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(message) > 5000:
        raise HTTPException(status_code=400, detail="Message too long")

    sid, session = get_or_create_session(session_id)
    system_prompt, classification, calc_result = run_pipeline(message.strip(), session)

    # Формируем данные калькулятора для UI
    import json
    calc_data = None
    if calc_result:
        calc_data = {
            "scenario": calc_result.scenario,
            "name": calc_result.name,
            "budget": calc_result.budget,
            "revenue_y1": calc_result.revenue_y1,
            "revenue_y2": calc_result.revenue_y2,
            "payback_months": calc_result.payback_months,
            "roi_24m": calc_result.roi_24m,
            "npv_3y": calc_result.npv_3y,
            "total_losses": calc_result.total_losses,
            "net_effect_y1": calc_result.net_effect_y1,
            "precision": calc_result.precision,
            "cfo_acceptable": calc_result.cfo_acceptable,
        }

    session.messages.append({"role": "user", "content": message.strip()})
    trimmed = trim_history(session.messages)
    messages = [{"role": "system", "content": system_prompt}] + trimmed

    async def event_generator():
        # Сначала отправляем classification + calc_result
        meta = {
            "session_id": sid,
            "classification": classification.category,
            "scenario": session.state.current_scenario,
            "calc": calc_data,
        }
        yield {"event": "meta", "data": json.dumps(meta, ensure_ascii=False)}

        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                max_tokens=512,
                messages=messages,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield {"event": "message", "data": text}

            session.messages.append({"role": "assistant", "content": full_response})
            session.messages = trim_history(session.messages)

            yield {"event": "done", "data": ""}

        except Exception as e:
            if session.messages and session.messages[-1]["role"] == "user":
                session.messages.pop()
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())


@app.get("/api/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    """Инспекция состояния сессии (для демо и дебага)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    s = sessions[session_id].state
    return {
        "scenario": s.current_scenario,
        "confidence": s.confidence,
        "pressure_count": s.pressure_count,
        "known_facts": s.known_facts,
        "position_history": [{"scenario": h[0], "reason": h[1]} for h in s.position_history],
        "messages_count": s.messages_count,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=422,
        content={
            "response": "Произошла ошибка обработки. Переформулируйте вопрос.",
            "error": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)

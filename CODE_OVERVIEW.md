# TriAgent - Code Overview

---

## 1. Key components

| Component                                                           | Implementation                                                                                                                                    |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shared process state                                                | `TriAgentState` TypedDict in [orchestrator.py:18–35](backend/orchestrator.py#L18-L35)                                                             |
| Knowledge Agent                                                     | `KnowledgeAgent` class in [agents/knowledge_agent.py](backend/agents/knowledge_agent.py)                                                          |
| Reasoning Agent                                                     | `ReasoningAgent` class in [agents/reasoning_agent.py](backend/agents/reasoning_agent.py)                                                          |
| Execution Agent                                                     | `ExecutionAgent` class in [agents/execution_agent.py](backend/agents/execution_agent.py)                                                          |
| Inter-agent protocol (knowledge/action request, result, escalation) | LangGraph graph in `_build_graph` ([orchestrator.py:52–80](backend/orchestrator.py#L52-L80))                                                      |
| FAISS rule store, similarity threshold θ = 0.75                     | `FAISSRuleStore` in [database/faiss_store.py](backend/database/faiss_store.py); `SIMILARITY_THRESHOLD = 0.75` at line 15                          |
| `all-MiniLM-L6-v2` embeddings                                       | `EMBEDDING_MODEL = "all-MiniLM-L6-v2"` in [faiss_store.py:14](backend/database/faiss_store.py#L14)                                                |
| ReAct loop with JSON output, confidence score                       | `SYSTEM_PROMPT` + loop `for step_num in range(1, 20)` in [reasoning_agent.py:52–133](backend/agents/reasoning_agent.py#L52-L133)                  |
| Confidence threshold θ_c = 0.6 → human-in-the-loop                  | `CONFIDENCE_THRESHOLD = 0.6` in [reasoning_agent.py:7](backend/agents/reasoning_agent.py#L7); check at line 88                                    |
| Retry-and-escalate (max 2 retries)                                  | `MAX_RETRIES = 2` in [execution_agent.py:7](backend/agents/execution_agent.py#L7); loop `range(1, MAX_RETRIES + 2)` at line 35 → 3 total attempts |
| Max 10 reasoning-execution cycles → escalation                      | check `tool_iterations >= 10` in `_routing` ([orchestrator.py:151](backend/orchestrator.py#L151))                                                 |

---

## 2. End-to-end execution flow

### 2.1 State-machine diagram

```
POST /process
     │
     ▼
orchestrator.process(request)           [orchestrator.py:160–237]
     │
     ├─► pre-fetch live weather (if location present)   [lines 173–175]
     │
     └─► graph.invoke(initial_state)
              │
              ▼
         ┌──────────────────────────────────────────────┐
         │           LangGraph state machine            │
         │                                              │
         │   ┌──────────┐                               │
         │   │knowledge │  _knowledge_node              │
         │   │  node    │  → KnowledgeAgent             │
         │   │          │    .handle_knowledge_request  │
         │   └────┬─────┘                               │
         │        │ always                              │
         │        ▼                                     │
         │   ┌──────────┐                               │
         │   │reasoning │  _reasoning_node              │
         │   │  node    │  → ReasoningAgent             │
         │   │          │    .run()  or  .resume_after_tool()
         │   └─────┬────┘                               │
         │         │                                    │
         │    ┌────┴───────────────────┐                │
         │    │  _routing()            │                │
         │    │  • "execute" → exec    │                │
         │    │  • "complete" → END    │                │
         │    │  • "escalate" → END    │                │
         │    └──────┬─────────────────┘                │
         │           │ "execute"                        │
         │           ▼                                  │
         │   ┌──────────────┐                           │
         │   │ execution    │  _execution_node          │
         │   │   node       │  → ExecutionAgent.execute │
         │   └──────┬───────┘                           │
         │          │ loop back                         │
         │          └─────────────► reasoning node      │
         └──────────────────────────────────────────────┘
              │
              ▼
         ProcessResult (JSON)
```

### 2.2 Step-by-step walkthrough (typical instance)

1. **Client** sends `POST /process?model=gpt` with body `{"description": "Orange pustules on wheat leaves...", "crop": "wheat", "location": "Rome, IT", "severity": "medium"}`.
2. **main.py:51** delegates to `orchestrators["gpt"].process(request)`.
3. **orchestrator.py:164–176**: builds the `context` dict (`crop`, `location`, `growth_stage`, `severity`). If `location` is present and `--skip-prefetch` is not set, calls `ExecutionAgent.execute("get_weather", ...)` to pre-fetch weather — added to `context["weather"]`.
4. **initial_state** is populated with all `None`/`[]`/`False` fields and `input_description = request.description`.
5. **Knowledge node** (`_knowledge_node`): `KnowledgeAgent.handle_knowledge_request(description)` → retrieves the most similar rule from the FAISS store (cosine similarity on trigger embedding). If `score >= 0.75`, returns the cached rule. Otherwise calls the LLM to generate a rule on-the-fly and adds it to the store. Result stored in `state["rule_data"]`.
6. **Reasoning node** (`_reasoning_node`, first visit): `ReasoningAgent.run(process_rule, input_description, context)` → sends the LLM a system prompt + extracted rule + field situation. LLM responds in structured JSON (thought/action/gateway_decision/confidence/requires_tool/...). The ReAct loop continues for up to 20 internal steps until `process_complete=True` or `requires_tool=True`.
7. If `requires_tool=True`: routing sends to the **execution node** with `tool_name` and `tool_params` in state.
8. **Execution node** (`_execution_node`): `ExecutionAgent.execute(tool_name, tool_params)` → calls the registered function (e.g. `get_weather(...)`). On failure: retry × 2 with linear backoff. If all retries fail: sets `escalate=True` → routing → `END` with `escalated=True`.
9. Tool result is stored in `state["tool_result"]`. Routing sends back to the reasoning node.
10. **Reasoning node** (subsequent visit): `ReasoningAgent.resume_after_tool(messages, tool_name, tool_result, ...)` → injects the tool result into the conversation history and continues the loop.
11. This cycle repeats until `complete=True` (or `escalated=True`, or `tool_iterations >= 10`).
12. **orchestrator.py:202–237**: builds `ProcessResult` from `final_state` with `success`, `diagnosis`, `treatment`, `actions_taken`, `reasoning_trace`, `rule_used`, `cost_usd`, `llm_model`.

---

## 3. Shared state schema (`TriAgentState`)

Defined in [orchestrator.py:18–35](backend/orchestrator.py#L18-L35):

| Field               | Type             | Set by                                | Description                                                                    |
| ------------------- | ---------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| `process_id`        | `str`            | `process()` (line 161)                | Short UUID for tracing                                                         |
| `input_description` | `str`            | `process()` (line 183)                | Free-text field situation                                                      |
| `context`           | `dict`           | `process()` (lines 164–176)           | crop, location, severity, growth_stage, weather (if available)                 |
| `rule_data`         | `Optional[dict]` | `_knowledge_node`                     | rule_id, trigger, content (rule text), similarity_score, source                |
| `reasoning_result`  | `Optional[dict]` | `_reasoning_node`                     | Raw output of `ReasoningAgent.run` or `resume_after_tool`                      |
| `tool_name`         | `Optional[str]`  | `_reasoning_node`                     | Tool name requested by reasoning (e.g. `"get_weather"`)                        |
| `tool_params`       | `Optional[dict]` | `_reasoning_node`                     | Tool parameters (e.g. `{"location": "Rome, IT"}`)                              |
| `tool_result`       | `Optional[str]`  | `_execution_node`                     | Serialised tool output; reset to `None` after each reasoning visit             |
| `messages`          | `List[dict]`     | `_reasoning_node`                     | LLM conversation history (system + user + assistant turns)                     |
| `actions_taken`     | `List[str]`      | `_reasoning_node`                     | Actions taken (the `action` field of each ReAct step)                          |
| `reasoning_trace`   | `List[dict]`     | `_reasoning_node`                     | Full ReAct steps (thought, action, observation, confidence, gateway_decision)  |
| `diagnosis`         | `Optional[str]`  | `_reasoning_node`                     | Final diagnosis extracted from the last step with a non-null `diagnosis` field |
| `treatment`         | `Optional[str]`  | `_reasoning_node`                     | Recommended treatment                                                          |
| `escalated`         | `bool`           | `_reasoning_node` / `_execution_node` | True if confidence < 0.6 or tool failure after all retries                     |
| `complete`          | `bool`           | `_reasoning_node`                     | True when `process_complete=True` in LLM response                              |
| `tool_iterations`   | `int`            | `_execution_node`                     | Reasoning-execution cycle counter; escalates if >= 10                          |
| `error`             | `Optional[str]`  | Unused (always None)                  | Placeholder for catastrophic errors                                            |

---

## 4. Knowledge Agent

File: [agents/knowledge_agent.py](backend/agents/knowledge_agent.py)

### Offline mode — rule generation from FAO/CIMMYT documents

`load_fao_documents(documents_path)` (line 96) reads PDFs with PyMuPDF, chunks them at 1000 words (`_chunk_text`), and for each chunk > 200 characters calls `generate_rule_from_document`. This uses the LLM with `RULE_GENERATION_PROMPT` to extract a structured rule:

```
<<trigger>>: [a phrase describing when the rule applies]
<<steps>>:
1. [action]
   <<decision>>: [condition]
     <<if yes>>: [action]
     <<if no>>: [action]
2. ...
<<end>>
```

The trigger is extracted from the generated rule (`_extract_trigger`) and used as the FAISS embedding key.

The pre-built FAISS index (`data/rules/faiss.index` + `data/rules/rules.json`) is included in the repository and is ready to use without regeneration.

### Online mode — similarity retrieval

`handle_knowledge_request(description)` (line 89):

1. `rule_store.retrieve(description)` — embeds the description and finds the nearest trigger in the FAISS index.
2. If `similarity_score >= 0.75` → returns the cached rule (`source = "retrieved"`).
3. If below threshold → calls `generate_rule_from_description` with `RULE_GENERATION_UNSTRUCTURED_PROMPT` → generates a rule on-the-fly, adds it to the store, returns it (`source = "generated"`).

The `source` field distinguishes retrieval hits from on-the-fly generation; instances where the KA generates on-the-fly are harder and more expensive.

---

## 5. Reasoning Agent

File: [agents/reasoning_agent.py](backend/agents/reasoning_agent.py)

### System prompt

`SYSTEM_PROMPT` (lines 17–41) instructs the LLM to respond **exclusively** in JSON:

```json
{
  "step": <int>,
  "thought": "<step-by-step reasoning>",
  "action": "<what to do or null>",
  "gateway_decision": "<yes/no/null>",
  "confidence": 0.0–1.0,
  "process_complete": <bool>,
  "requires_tool": <bool>,
  "tool_name": "<name or null>",
  "tool_params": {},
  "diagnosis": "<final diagnosis or null>",
  "treatment": "<recommendation or null>"
}
```

The deterministic JSON format makes reasoning machine-parseable and traceable, enabling automatic ablation verification.

### Main loop (`run`, line 52)

```
initialise messages with [system_prompt, user_message(rule + situation + context)]
for step_num in 1..19:
    call LLM (temperature=0, response_format=json_object)
    parse JSON response
    if confidence < 0.6: escalate and break
    update diagnosis / treatment / actions_taken
    if process_complete: break
    if requires_tool: return {status:"tool_required", tool_name, tool_params, messages, ...}
    append response to messages; add "Continue to the next step."
return {status:"complete"/"escalated", reasoning_trace, diagnosis, treatment, ...}
```

### Resume after tool call (`resume_after_tool`, line 135)

Resumes from the already-built conversation history (passed through state), injects the tool result as a user message `"'{tool_name}' returned: {tool_result}\nContinue."`, and continues the loop. This preserves multi-turn context without re-sending the full rule to the LLM.

### Token counting

`self.total_tokens += resp.usage.total_tokens` on every call (lines 70, 149). Accumulated for cost calculation in `orchestrator.process`.

---

## 6. Execution Agent

File: [agents/execution_agent.py](backend/agents/execution_agent.py)

### Tool registry

6 tools registered at construction (`_register_default_tools`, line 17):

| Tool name                | Function                          | Type                     | Description                                                                                               |
| ------------------------ | --------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------- |
| `get_weather`            | `tools.weather_api.get_weather`   | **Live API**             | Calls OpenWeatherMap HTTP API; returns temperature, humidity, conditions                                  |
| `get_disease_info`       | `tools.eppo_api.get_disease_info` | **Live API + fallback**  | Calls EPPO Global Database; falls back to a local dict on timeout                                         |
| `send_sms`               | `tools.eppo_api.send_sms`         | **Twilio or simulation** | Sends SMS via Twilio; simulates if `TWILIO_ACCOUNT_SID` is not set                                        |
| `get_treatment_protocol` | `self._get_treatment_protocol`    | **Hard-coded dict**      | Protocols for wheat_brown_rust / tomato_late_blight / olive_fruit_fly × low/medium/high                   |
| `calculate_dosage`       | `self._calculate_dosage`          | **Arithmetic**           | Dose per area = dose_per_ha × area_ha                                                                     |
| `check_waiting_period`   | `self._check_waiting_period`      | **Hard-coded dict**      | Pre-harvest intervals for tebuconazole / propiconazole / mancozeb / metalaxyl / dimethoate / deltamethrin |

### Linear-backoff retry

The loop in `execute` (line 35) does `range(1, MAX_RETRIES + 2)` = 3 attempts: attempt 1, sleep(0.5 s), attempt 2, sleep(1.0 s), attempt 3. If all fail: `{success: False, escalate: True}`. The paper states "retry up to two times" — consistent (2 retries = 3 total attempts).

---

## 7. FAISS Rule Store

File: [database/faiss_store.py](backend/database/faiss_store.py)

- `IndexFlatIP` on L2-normalised vectors = exact cosine similarity (not approximate).
- NumPy fallback if `faiss` is not installed (dot product on embedding matrix).
- `save(path)` / `load(path)`: serialises rules to `rules.json` (trigger + content + metadata) and index to `faiss.index`. The pre-built index is included in `data/rules/`.

---

## 8. Cost accounting

Cost is calculated in [orchestrator.py:210–215](backend/orchestrator.py#L210-L215):

```python
total_tokens = self.knowledge_agent.total_tokens + self.reasoning_agent.total_tokens
cost = round((total_tokens / 1_000_000) * COST_PER_1M_TOKENS, 4)
```

A per-backbone cost table is defined in `backend/config/settings.py` (`DEFAULT_COST_TABLE`), replacing the earlier single hard-coded constant. Local models (gemma4-local, qwen-local) have cost 0.

---

## 9. Local backbones via llama.cpp server

The llama.cpp server exposes `/v1/chat/completions` with the same interface as the OpenAI API. Integration requires only pointing the OpenAI SDK client to a different `base_url`:

```python
from openai import OpenAI
client = OpenAI(api_key="lcpp", base_url="http://localhost:8080/v1")
# "model" in the request must match the alias passed with -a to the server
resp = client.chat.completions.create(model="gemma4-local", messages=[...], temperature=0.0)
```

The `api_key` can be any string (the server does not validate it). No agent code changes are needed — `make_client(provider)` returns the OpenAI client with the correct `base_url`.

**Caveat**: some GGUFs do not honour `response_format={"type": "json_object"}`. For these, `ReasoningAgent` falls back to `json.loads(raw)` and, on `JSONDecodeError`, extracts the first `{...}` block with `re.search(r'\{[\s\S]*\}', raw)`. The `--reasoning off` flag in the llama.cpp server command disables `<think>` tags that would interfere with JSON parsing.

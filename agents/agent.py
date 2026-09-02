"""
LangGraph agentic workflow for the mode_choice project.

Takes natural-language input from a user, routes it through a Groq-hosted LLM,
and gives that LLM two tool surfaces:

  1. A Postgres SQL toolkit (list tables / inspect schema / run queries / check
     queries) built by LangChain's SQLDatabaseToolkit, for question-answering
     over data living in Postgres.
  2. A `use_xgboost_model` tool that wraps this project's XGBoost mode-choice
     classifier. The exact calling convention (which fields the LLM should
     extract from the conversation, in what order) isn't finalized yet, so
     this tool is a working placeholder: it accepts an arbitrary JSON object
     of feature_name -> value pairs, builds a one-row frame, and calls
     .predict / .predict_proba on whatever model is found at
     XGBOOST_MODEL_PATH. Tighten `EXPECTED_FEATURES` once the training
     pipeline's feature schema (see src/data/data_preprocess.py) is final.

Graph shape (classic LangGraph ReAct loop, built from scratch rather than
langgraph.prebuilt.create_react_agent, per project requirements):

    START -> agent -> [should_continue] -> action -> agent -> ... -> END
                              |
                              +-> END (once the model stops calling tools)

Run directly for an interactive NL REPL:

    python agents/agent.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# --------------------------------------------------------------------------
# Config -- placeholders. Fill these in via a .env file (see .env.example)
# or environment variables; nothing here is a real credential.
# --------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "PLACEHOLDER_GROQ_API_KEY")

# Pick any Groq-hosted model id once you've decided, e.g. "llama-3.3-70b-versatile"
# or "openai/gpt-oss-120b". Left as a placeholder deliberately.
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "PLACEHOLDER_MODEL_NAME")

# SQLAlchemy connection string for the Postgres database the SQL tool queries.
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgresql+psycopg2://user:password@localhost:5432/mode_choice"
)

# Where the trained XGBoost mode-choice model lives. train_pipeline.py currently
# only persists the NN models under artifacts/nn_models/ -- point this at your
# saved XGBoost model (joblib dump or native .json Booster) once you train one.
XGBOOST_MODEL_PATH = os.environ.get(
    "XGBOOST_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "artifacts" / "xgb_model.json")
)


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)


# --------------------------------------------------------------------------
# SQL tool (Postgres)
# --------------------------------------------------------------------------

db = SQLDatabase.from_uri(POSTGRES_URI)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()
# -> [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]


# --------------------------------------------------------------------------
# XGBoost model tool (placeholder invocation contract)
# --------------------------------------------------------------------------

_xgb_model_cache: dict = {}


def _load_xgboost_model():
    """Lazily load and cache the XGBoost model from XGBOOST_MODEL_PATH."""
    if "model" in _xgb_model_cache:
        return _xgb_model_cache["model"]

    model_path = Path(XGBOOST_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No XGBoost model found at {model_path}. Train one (see "
            "train_pipeline.py / src/training/train.py) and set XGBOOST_MODEL_PATH."
        )

    if model_path.suffix in {".json", ".ubj"}:
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(model_path))
    else:
        import joblib

        model = joblib.load(model_path)

    _xgb_model_cache["model"] = model
    return model


@tool
def use_xgboost_model(features_json: str) -> str:
    """Run the trained XGBoost mode-choice model on a single record.

    Args:
        features_json: A JSON object string mapping feature names to values
            for one trip/record, e.g. '{"trip_distance": 4.2, "age": 34, ...}'.
            The exact required fields depend on the feature schema the model
            was trained on (see src/data/data_preprocess.py:process_data_xgboost)
            -- pull those from the conversation or from a prior SQL tool call
            before calling this tool.

    Returns:
        A JSON string with the predicted class (and class probabilities, if
        the loaded model supports predict_proba) or an error message if no
        trained model is available yet.
    """
    import pandas as pd

    try:
        model = _load_xgboost_model()
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})

    try:
        record = json.loads(features_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"features_json was not valid JSON: {exc}"})

    row = pd.DataFrame([record])

    try:
        if hasattr(model, "predict_proba"):  # sklearn-style XGBClassifier
            proba = model.predict_proba(row)[0]
            prediction = int(proba.argmax())
            return json.dumps({"prediction": prediction, "probabilities": proba.tolist()})

        import xgboost as xgb  # native Booster

        dmatrix = xgb.DMatrix(row)
        raw = model.predict(dmatrix)
        return json.dumps({"prediction": raw.tolist()})
    except Exception as exc:  # model/feature mismatch, dtype issues, etc.
        return json.dumps({"error": f"Prediction failed: {exc}"})


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------

tools = [*sql_tools, use_xgboost_model]
tools_by_name = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are the mode_choice project's assistant. You can inspect and query "
        "a Postgres database (list tables, view schema, run read queries) and, "
        "when you have gathered a full set of feature values for a record, call "
        "the use_xgboost_model tool to get a mode-choice prediction. Always check "
        "the table schema before writing a query, and never guess column names."
    )
)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def call_model(state: AgentState) -> dict:
    """Ask the LLM for its next message, given the conversation so far."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT, *messages]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def execute_tools(state: AgentState) -> dict:
    """Run every tool call requested by the last AI message and collect results."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = tools_by_name.get(tool_name)

        if selected_tool is None:
            result = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                result = selected_tool.invoke(tool_args)
            except Exception as exc:
                result = f"Error running tool '{tool_name}': {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
        )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Route to the tool-execution node if the model asked for tools, else end."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"


# --------------------------------------------------------------------------
# Build + compile the graph
# --------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", execute_tools)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------
# Convenience entry points
# --------------------------------------------------------------------------


def run_agent(user_input: str, thread_id: str = "default-thread") -> str:
    """Run one NL turn through the graph and return the assistant's reply text."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    return result["messages"][-1].content


def main() -> None:
    print("mode_choice agent -- type your question ('exit' to quit).")
    print(f"model={GROQ_MODEL_NAME!r} db={POSTGRES_URI!r}")

    thread_id = "cli-session"
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        reply = run_agent(user_input, thread_id=thread_id)
        print(f"agent> {reply}")


if __name__ == "__main__":
    main()

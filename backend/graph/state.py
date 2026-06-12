# Location: backend/graph/state.py
from typing import TypedDict, Any

CHAT       = "chat"
RAG        = "rag"
POSTMORTEM = "postmortem"


class OpsState(TypedDict):
    mode:             str
    user_input:       str
    file_path:        str
    response:         str
    chat_memory:      Any
    rag_store:        Any
    rag_memory:       Any
    pm_store:         Any
    pm_memory:        Any
    report_str:       str
    llm:              Any
    rag_warning:      str
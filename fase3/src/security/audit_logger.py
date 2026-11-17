"""
Logging estruturado em JSON para auditoria de todas as interações do assistente.
Cada entrada registra: timestamp, session_id, pergunta, resposta, fontes e flags de segurança.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


class AuditLogger:
    def __init__(self, log_path: str = "logs/assistant.jsonl"):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def log(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[str],
        safety_flags: list[str],
        node: str = "assistant",
        patient_id: str | None = None,
    ) -> str:
        entry_id = str(uuid.uuid4())[:8]
        entry = {
            "entry_id": entry_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "node": node,
            "patient_id": patient_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "safety_flags": safety_flags,
            "requires_human_review": len(safety_flags) > 0,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry_id

    def get_session_logs(self, session_id: str) -> list[dict]:
        logs = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("session_id") == session_id:
                        logs.append(entry)
        except FileNotFoundError:
            pass
        return logs

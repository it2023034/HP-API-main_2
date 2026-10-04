import re
from typing import List, Dict, Any
from collections import defaultdict

def extract_date_str(timestamp_val: Any) -> str:
    """Extracts date string (YYYY-MM-DD or DD/MM/YYYY) from timestamp field."""
    if not timestamp_val or str(timestamp_val).strip() == "":
        return "UNKNOWN_DATE"
    
    val_str = str(timestamp_val).strip()
    return val_str.split(" ")[0].split("T")[0]


def create_chunks_by_date_and_window(
    messages: List[Dict[str, Any]], 
    max_messages_per_chunk: int = 15,
    overlap: int = 2
) -> List[List[Dict[str, Any]]]:
    """Groups messages strictly by date boundaries and applies sliding window if needed."""
    if not messages:
        return []

    grouped_by_date = defaultdict(list)
    for msg in messages:
        raw_ts = msg.get("Date", msg.get("Timestamp", msg.get("timestamp", "")))
        date_key = extract_date_str(raw_ts)
        grouped_by_date[date_key].append(msg)

    final_chunks: List[List[Dict[str, Any]]] = []
    step = max_messages_per_chunk - overlap

    for date_key, date_messages in grouped_by_date.items():
        total_msgs = len(date_messages)
        
        if total_msgs <= max_messages_per_chunk:
            final_chunks.append(date_messages)
        else:
            for start_idx in range(0, total_msgs, step):
                sub_chunk = date_messages[start_idx : start_idx + max_messages_per_chunk]
                if sub_chunk:
                    final_chunks.append(sub_chunk)
                if start_idx + max_messages_per_chunk >= total_msgs:
                    break

    return final_chunks


def format_chunk_to_text(chunk: List[Dict[str, Any]]) -> str:
    """Formats a message chunk into a clean string for LLM prompting."""
    formatted_lines = []
    for msg in chunk:
        sender = msg.get("Sender", msg.get("sender", "Unknown"))
        receiver = msg.get("Receiver", msg.get("receiver", "Unknown"))
        content = msg.get("Message", msg.get("content", ""))
        timestamp = msg.get("Timestamp", msg.get("timestamp", msg.get("Date", "")))

        if timestamp:
            formatted_lines.append(f"[{timestamp}] {sender} -> {receiver}: {content}")
        else:
            formatted_lines.append(f"{sender} -> {receiver}: {content}")

    return "\n".join(formatted_lines)
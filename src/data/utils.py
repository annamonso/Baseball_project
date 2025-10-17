from datetime import datetime, timedelta
from typing import List, Tuple

def date_chunks(start: str, end: str, chunk_days: int) -> List[Tuple[str,str]]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=chunk_days-1), e)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + timedelta(days=1)
    return out
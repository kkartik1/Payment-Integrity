import re

def split_sections(named_blocks: list[tuple[str, str]], min_tokens=500, max_tokens=1200):
    """
    named_blocks: list of (section_name, plain_text)
    Returns list of (section_path, chunk_text)
    """
    chunks = []
    for sec_name, txt in named_blocks:
        if not txt: 
            continue
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        cur, cur_len = [], 0
        for p in paras:
            tokens = len(p.split())
            if cur_len + tokens > max_tokens and cur:
                chunks.append((f"{sec_name} → part {len(chunks)+1}", "\n\n".join(cur)))
                cur, cur_len = [p], tokens
            else:
                cur.append(p)
                cur_len += tokens
        if cur:
            chunks.append((f"{sec_name} → part {len(chunks)+1}", "\n\n".join(cur)))
    # Merge smaller tails where needed
    merged = []
    buf = None
    for sec, txt in chunks:
        if buf is None:
            buf = [sec, txt]
            continue
        if len(buf[1].split()) < min_tokens:
            buf[1] = buf[1] + "\n\n" + txt
        else:
            merged.append(tuple(buf))
            buf = [sec, txt]
    if buf: merged.append(tuple(buf))
    return merged
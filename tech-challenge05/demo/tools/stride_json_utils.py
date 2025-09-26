# tools/stride_json_utils.py
import re, json
from typing import Any, Dict, Optional

def _strip_code_fences(s: str) -> str:
    # remove ```json ... ``` etc.
    return re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), s)

def extract_json_block(text: str) -> Optional[str]:
    """
    Acha o primeiro objeto JSON balanceado no texto (contando chaves),
    ignorando qualquer frase antes/depois.
    """
    s = _strip_code_fences(text)
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def try_load_json(raw: str) -> Dict[str, Any]:
    block = extract_json_block(raw)
    if not block:
        raise ValueError("Nenhum bloco JSON encontrado.")
    # pequenos reparos comuns (vírgulas no fim de arrays/objetos)
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    return json.loads(block)

def coerce_stride_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garante chaves esperadas e tipos básicos.
    """
    out = {
        "stride_summary": obj.get("stride_summary") or "",
        "threats": obj.get("threats") or {},
        "overall_risk": (obj.get("overall_risk") or "unknown").lower(),
        "top_mitigations": obj.get("top_mitigations") or []
    }
    # normaliza estrutura de threats: dict[stride] -> list[{component,reason,mitigations}]
    norm = {}
    if isinstance(out["threats"], dict):
        for k, v in out["threats"].items():
            items = []
            if isinstance(v, list):
                for it in v:
                    if not isinstance(it, dict): 
                        continue
                    items.append({
                        "component": it.get("component",""),
                        "reason": it.get("reason",""),
                        "mitigations": it.get("mitigations") or []
                    })
            norm[k] = items
    out["threats"] = norm
    # lista de strings p/ top_mitigations
    out["top_mitigations"] = [str(x) for x in out["top_mitigations"] if x]
    return out

def parse_stride_json_loose(raw_text: str) -> Dict[str, Any]:
    """
    Pipeline completo: extrai JSON do texto bruto e adapta pro schema.
    Levanta ValueError se não conseguir.
    """
    obj = try_load_json(raw_text)
    return coerce_stride_schema(obj)

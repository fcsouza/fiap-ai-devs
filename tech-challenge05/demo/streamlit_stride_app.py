import os, io, re, json, time, difflib, tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np
import yaml
import cv2
import pytesseract
import builtins
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from tools.stride_json_utils import parse_stride_json_loose

# -------------- Config --------------

DEFAULT_CFG = {
    "yolo": {"model_path":"demo/models/best.pt","data_yaml":"demo/data.yaml","imgsz":832,"conf":0.35,"iou":0.60,"provider_filter":"Todos"},
    "ocr": {"lang":"eng","min_conf":70,"min_len":3,"stopwords":["the","and","api","rest","soap","http"],"dedupe_sim":0.90,"join_nearby_px":16},
    "stride": {"llm_model":"llama3","temperature":0.0,"top_p":0.1,"max_texts":40,"max_icons":80},
    "report": {"logo_path":None,"author":"STRIDE Analyzer","title":"Relatório STRIDE"}
}

@st.cache_data(show_spinner=False)
def load_yaml(path:str)->dict:
    with open(path,"r") as f:
        return yaml.safe_load(f)

def load_config(path="stride_app_config.yaml")->dict:
    if os.path.exists(path):
        try:
            cfg = load_yaml(path)
        except Exception:
            cfg = DEFAULT_CFG
    else:
        cfg = DEFAULT_CFG
    # fill defaults if keys missing
    def deepfill(base, add):
        for k,v in add.items():
            if k not in base:
                base[k] = v
            elif isinstance(v, dict):
                base[k] = deepfill(base.get(k,{}), v)
        return base
    return deepfill(cfg, DEFAULT_CFG)

CFG = load_config()

# -------------- Helpers --------------

def read_names(data_yaml:str)->List[str]:
    with open(data_yaml,"r") as f:
        return yaml.safe_load(f)["names"]

def compute_provider_index(names:List[str])->Dict[str, List[int]]:
    return {
        "AWS":   [i for i,n in enumerate(names) if n.startswith("aws_") or n.startswith("aws_amazon_")],
        "Azure": [i for i,n in enumerate(names) if n.startswith("azure_") or n.startswith("microsoft_entra")],
        "GCP":   [i for i,n in enumerate(names) if n.startswith("gcp_")],
        "API":   [i for i,n in enumerate(names) if n=="api"],
        "Todos": list(range(len(names))),
    }

def pil_to_cv(img:Image.Image)->np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr:np.ndarray)->Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

# perto do topo do arquivo (ou onde já estava sua função iou)
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

# -------------- YOLO --------------

@st.cache_resource
def load_model(path:str):
    return YOLO(path)

@st.cache_data(show_spinner=False)
def detect_icons(image:Image.Image, yolo_model_path:str, data_yaml:str, imgsz:int, conf:float, iou_thr:float, provider_filter:str):
    model = load_model(yolo_model_path)
    names = read_names(data_yaml)
    prov_map = compute_provider_index(names)
    classes = prov_map.get(provider_filter, prov_map["Todos"])

    res = model.predict(
        source=pil_to_cv(image),
        imgsz=imgsz,
        conf=conf,
        iou=iou_thr,
        agnostic_nms=False,
        max_det=1000,
        classes=classes if provider_filter!="Todos" else None,
        verbose=False
    )[0]

    # tabela
    dets = []
    if res.boxes is not None and len(res.boxes)>0:
        for i in range(len(res.boxes)):
            x1,y1,x2,y2 = [float(v) for v in res.boxes.xyxy[i].tolist()]
            cls_id = int(res.boxes.cls[i].item())
            c = float(res.boxes.conf[i].item())
            dets.append({
                "cls_id": cls_id,
                "label": names[cls_id],
                "conf": c,
                "bbox": [x1,y1,x2,y2],
            })
    # imagem anotada
    plotted = res.plot()  # BGR
    return dets, cv_to_pil(plotted), names

# -------------- OCR --------------

def merge_nearby_boxes(boxes, join_px=16):
    """Junta caixas próximas (p.ex. linha quebrada) e concatena texto."""
    merged = []
    used = set()
    for i,a in enumerate(boxes):
        if i in used: continue
        ax1,ay1,ax2,ay2 = a["bbox"]; txtA = a["text"]
        group = [i]
        for j,b in enumerate(boxes[i+1:], start=i+1):
            bx1,by1,bx2,by2 = b["bbox"]
            # proximidade horizontal/vertical
            close = (abs(by1 - ay2) <= join_px) or (abs(ay1 - by2) <= join_px) or (abs(bx1-ax2)<=join_px) or (abs(ax1-bx2)<=join_px)
            if close or box_iou([ax1,ay1,ax2,ay2],[bx1,by1,bx2,by2])>0.15:
                group.append(j)
        used.update(group)
        xs = [boxes[k]["bbox"][0] for k in group] + [boxes[k]["bbox"][2] for k in group]
        ys = [boxes[k]["bbox"][1] for k in group] + [boxes[k]["bbox"][3] for k in group]
        merged_bbox = [min(xs), min(ys), max(xs), max(ys)]
        merged_txt = " ".join([boxes[k]["text"] for k in group])
        merged.append({"bbox": merged_bbox, "text": merged_txt})
    return merged

def normalize_text(t:str)->str:
    t = t.strip()
    t = re.sub(r"[^\w\s\-\.:/]+","", t)
    t = re.sub(r"\s+"," ", t)
    return t

@st.cache_data(show_spinner=False)
def ocr_text(image: Image.Image, lang="eng", min_conf=70, min_len=3, stopwords=None, dedupe_sim=0.9, join_nearby_px=16):
    stopwords = set(stopwords or [])
    img = pil_to_cv(image)
    data = pytesseract.image_to_data(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    boxes = []
    for i, txt in enumerate(data.get("text", [])):
        txt2 = normalize_text(txt or "")
        if len(txt2) < int(min_len):
            continue
        if txt2.lower() in stopwords:
            continue
        try:
            conf_val = data.get("conf", [])[i]
            conf = builtins.float(conf_val)
        except Exception:
            conf = -1.0
        if conf < builtins.float(min_conf):
            continue
        x = int(data["left"][i]); y = int(data["top"][i])
        w = int(data["width"][i]); h = int(data["height"][i])
        boxes.append({"bbox": [x, y, x + w, y + h], "text": txt2, "conf": conf})

    boxes = merge_nearby_boxes(boxes, join_px=int(join_nearby_px))

    uniq = []
    for b in boxes:
        t = b["text"]
        if any(difflib.SequenceMatcher(None, t.lower(), u["text"].lower()).ratio() >= builtins.float(dedupe_sim) for u in uniq):
            continue
        uniq.append(b)

    overlay = img.copy()
    for b in uniq:
        x1, y1, x2, y2 = map(int, b["bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 160, 255), 2)
        cv2.putText(overlay, b["text"][:24], (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 160, 255), 1)

    return uniq, cv_to_pil(overlay)


# -------------- Consolidação (ícones + textos) --------------

def summarize_icons(dets:List[dict])->Dict[str, Any]:
    by_label = {}
    for d in dets:
        lab = d["label"]
        by_label.setdefault(lab, {"count":0, "conf": []})
        by_label[lab]["count"] += 1
        by_label[lab]["conf"].append(d["conf"])
    summary=[]
    for lab, v in by_label.items():
        summary.append({
            "label": lab,
            "count": v["count"],
            "mean_conf": float(np.mean(v["conf"])) if v["conf"] else 0.0
        })
    summary.sort(key=lambda x:(-x["count"], -x["mean_conf"]))
    return {"total": len(dets), "by_label": summary}

def summarize_texts(text_boxes:List[dict], max_texts:int)->List[str]:
    # prioriza termos “grandes” e com dígitos/palavras compostas
    def score(t):
        s = t["text"]
        return len(s) + (2 if any(ch.isdigit() for ch in s) else 0) + (1 if "_" in s or "-" in s else 0)
    ranked = sorted(text_boxes, key=score, reverse=True)
    uniq = []
    seen = set()
    for r in ranked:
        k = r["text"].strip().lower()
        if k in seen: continue
        seen.add(k); uniq.append(r["text"])
        if len(uniq)>=max_texts: break
    return uniq

# -------------- STRIDE (Ollama) --------------

def call_ollama_stride(model:str, prompt:str, temperature:float, top_p:float)->str:
    try:
        import ollama
        resp = ollama.chat(
            model=model,
            messages=[{"role":"user","content":prompt}],
            options={"temperature":temperature, "top_p":top_p}
        )
        return resp["message"]["content"]
    except Exception as e:
        return json.dumps({
            "error": f"Ollama indisponível: {e}",
            "stride_summary": "fallback heurístico",
            "threats": {
                "spoofing": [],
                "tampering": [],
                "repudiation": [],
                "information_disclosure": [],
                "denial_of_service": [],
                "elevation_of_privilege": []
            },
            "overall_risk": "medium"
        }, ensure_ascii=False)

def build_stride_prompt(icon_summary:dict, texts:List[str], image_meta:dict)->str:
    # Instruímos o LLM a responder em JSON estruturado
    return f"""
Você é um analista de segurança. Dado um diagrama de arquitetura, use STRIDE para identificar ameaças.
Responda **EXCLUSIVAMENTE** em JSON com o schema abaixo (sem explicações fora do JSON).

Schema:
{{
  "stride_summary": "resumo em pt-br",
  "threats": {{
    "spoofing": [{{"component":"...", "reason":"...", "mitigations":["..."]}}],
    "tampering": [...],
    "repudiation": [...],
    "information_disclosure": [...],
    "denial_of_service": [...],
    "elevation_of_privilege": [...]
  }},
  "overall_risk": "low|medium|high",
  "top_mitigations": ["...", "..."]
}}

Contexto:
- Componentes detectados por ícone (com contagem e confiança média):
{json.dumps(icon_summary, ensure_ascii=False, indent=2)}

- Textos relevantes no diagrama (OCR):
{json.dumps(texts, ensure_ascii=False, indent=2)}

- Metadados da imagem:
{json.dumps(image_meta, ensure_ascii=False, indent=2)}

Regras:
- Cruce ícones + textos para inferir riscos (p.ex. "S3", "Public Subnet", "SQL", "Key Vault", "IAM", "API Gateway", "OpenAI", etc.).
- Evite redundâncias. Agregue ameaças similares no mesmo item.
- Dê exemplos de mitigação **práticos** (WAF, SG/NSG, KMS, IAM least-privilege, private endpoints, throttling, retries, audit logs, etc.).
- Mantenha a resposta **válida em JSON**.
"""

def analyze_stride(icon_dets:List[dict], text_boxes:List[dict], cfg:dict, image_size:Tuple[int,int])->dict:
    icon_summary = summarize_icons(icon_dets)
    texts = summarize_texts(text_boxes, max_texts=cfg["stride"]["max_texts"])
    image_meta = {"width": image_size[0], "height": image_size[1]}
    prompt = build_stride_prompt(icon_summary, texts, image_meta)
    raw = call_ollama_stride(cfg["stride"]["llm_model"], prompt, cfg["stride"]["temperature"], cfg["stride"]["top_p"])
    try:
        data = parse_stride_json_loose(raw)
        return {"prompt": prompt, "raw": raw, "json": data, "icon_summary": icon_summary, "texts": texts}
    except Exception:
        # tenta salvar bruto
        return {"prompt": prompt, "raw": raw, "json": None, "icon_summary": icon_summary, "texts": texts}

# -------------- PDF --------------

def make_pdf_report(pdf_path:str, original_img:Image.Image, vis_icons:Image.Image, vis_texts:Image.Image, stride_result:dict, cfg:dict):
    W, H = A4  # 595 x 842 pt
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow=[]
    title = cfg["report"]["title"] or "Relatório STRIDE"
    author = cfg["report"]["author"] or "Analyst"

    # Header
    if cfg["report"]["logo_path"] and os.path.exists(cfg["report"]["logo_path"]):
        flow.append(RLImage(cfg["report"]["logo_path"], width=120, height=40))
        flow.append(Spacer(1, 10))
    flow.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    flow.append(Paragraph(f"Autor: {author}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    # Mini sumário
    icon_sum = stride_result["icon_summary"]
    flow.append(Paragraph("<b>Resumo de ícones detectados</b>", styles["Heading3"]))
    rows = [["Componente","Qtd","Conf. média"]]
    for it in icon_sum["by_label"][:16]:
        rows.append([it["label"], str(it["count"]), f"{it['mean_conf']:.2f}"])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)
    ]))
    flow.append(t)
    flow.append(Spacer(1,10))

    # Imagens
    def img2rl(im:Image.Image, maxw=500):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        iw, ih = im.size
        scale = min(1.0, maxw/iw)
        return RLImage(buf, width=iw*scale, height=ih*scale)

    flow.append(Paragraph("<b>Diagrama (original)</b>", styles["Heading3"]))
    flow.append(img2rl(original_img))
    flow.append(Spacer(1,10))
    flow.append(Paragraph("<b>Detecções por ícones (YOLO)</b>", styles["Heading3"]))
    flow.append(img2rl(vis_icons))
    flow.append(Spacer(1,10))
    flow.append(Paragraph("<b>Textos detectados (OCR)</b>", styles["Heading3"]))
    flow.append(img2rl(vis_texts))
    flow.append(PageBreak())

    # STRIDE
    flow.append(Paragraph("<b>Análise STRIDE</b>", styles["Heading2"]))

    js = stride_result.get("json")
    raw = stride_result.get("raw")
    if js:
        # Overview
        flow.append(Paragraph("<b>Resumo</b>", styles["Heading3"]))
        flow.append(Paragraph(js.get("stride_summary","(sem resumo)"), styles["BodyText"]))
        flow.append(Spacer(1,8))
        flow.append(Paragraph(f"<b>Risco geral:</b> {js.get('overall_risk','(n/a)')}", styles["BodyText"]))
        flow.append(Spacer(1,8))

        # Tabelas por categoria
        for cat in ["spoofing","tampering","repudiation","information_disclosure","denial_of_service","elevation_of_privilege"]:
            flow.append(Paragraph(cat.replace("_"," ").title(), styles["Heading3"]))
            items = js.get("threats",{}).get(cat,[])
            if not items:
                flow.append(Paragraph("— sem achados —", styles["Italic"]))
                flow.append(Spacer(1,6))
                continue
            rows = [["Componente","Motivo","Mitigações"]]
            for it in items:
                rows.append([it.get("component","?"), it.get("reason","?"), "• " + "\n• ".join(it.get("mitigations",[]))])
            t = Table(rows, colWidths=[130, 260, 150])
            t.setStyle(TableStyle([
                ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                ("VALIGN",(0,0),(-1,-1),"TOP")
            ]))
            flow.append(t)
            flow.append(Spacer(1,8))

        top_mit = js.get("top_mitigations",[])
        if top_mit:
            flow.append(Paragraph("<b>Top Mitigações</b>", styles["Heading3"]))
            flow.append(Paragraph("• " + "<br/>• ".join(top_mit), styles["BodyText"]))
    else:
        flow.append(Paragraph("O LLM não retornou um JSON válido. Conteúdo bruto:", styles["BodyText"]))
        flow.append(Paragraph(f"<font size=8><pre>{raw}</pre></font>", styles["BodyText"]))

    doc.build(flow)

# -------------- UI --------------

st.set_page_config(page_title="STRIDE Analyzer", layout="wide")
st.title("🧠 STRIDE Analyzer (Ícones + OCR + LLM + PDF)")

with st.sidebar:
    st.subheader("Configuração")
    cfg_path = st.text_input("Config YAML", value="stride_app_config.yaml")
    if st.button("Recarregar config"):
        st.experimental_set_query_params(ts=str(time.time()))
    st.caption("Edite `stride_app_config.yaml` para salvar ajustes.")

cfg = CFG  # já carregado

tabs = st.tabs(["Config & Upload", "Validação Ícones", "Validação Textos", "Análise STRIDE", "Relatório (PDF)"])
state = st.session_state

# Session store
for k in ["orig_img","vis_icons","vis_texts","icon_dets","text_boxes","names","stride_result"]:
    if k not in state: state[k]=None

# --- TAB 1: Config & Upload ---
with tabs[0]:
    st.markdown("### Upload de imagem")
    up = st.file_uploader("Envie PNG/JPG do diagrama", type=["png","jpg","jpeg"])
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### YOLO")
        provider = st.selectbox("Filtro de provedor", ["Todos","AWS","Azure","GCP","API"], index=["Todos","AWS","Azure","GCP","API"].index(cfg["yolo"]["provider_filter"]))
        conf = st.slider("conf", 0.0, 1.0, float(cfg["yolo"]["conf"]), 0.01)
        nms_iou = st.slider("iou", 0.2, 0.95, float(cfg["yolo"]["iou"]), 0.01)
        imgsz = st.selectbox("imgsz", [640, 832, 896, 1024], index=[640,832,896,1024].index(int(cfg["yolo"]["imgsz"])))
    with colB:
        st.markdown("#### OCR")
        lang = st.text_input("Idioma (Tesseract)", value=cfg["ocr"]["lang"])
        min_conf = st.slider("Min conf OCR", 0, 100, int(cfg["ocr"]["min_conf"]), 1)
        dedupe_sim = st.slider("Dedupe (similaridade)", 0.7, 1.0, float(cfg["ocr"]["dedupe_sim"]), 0.01)
        join_px = st.slider("Juntar caixas próximas (px)", 0, 64, int(cfg["ocr"]["join_nearby_px"]), 1)

    if up:
        img = Image.open(io.BytesIO(up.read())).convert("RGB")
        state.orig_img = img
        st.image(img, caption="Original", use_column_width=True)
        st.success("Imagem carregada!")

    # salva ajustes no runtime (não persiste no arquivo)
    cfg["yolo"].update({"provider_filter":provider,"conf":float(conf),"iou":float(nms_iou),"imgsz":int(imgsz)})
    cfg["ocr"].update({"lang":lang,"min_conf":int(min_conf),"dedupe_sim":float(dedupe_sim),"join_nearby_px":int(join_px)})

# --- TAB 2: Validação Ícones (YOLO) ---
with tabs[1]:
    st.markdown("### Detecção de Ícones (YOLO)")
    if state.orig_img is None:
        st.info("Envie uma imagem na aba anterior.")
    else:
        with st.spinner("Detectando..."):
            icon_dets, vis, names = detect_icons(
                state.orig_img, cfg["yolo"]["model_path"], cfg["yolo"]["data_yaml"],
                cfg["yolo"]["imgsz"], cfg["yolo"]["conf"], cfg["yolo"]["iou"], cfg["yolo"]["provider_filter"]
            )
        state.icon_dets = icon_dets
        state.vis_icons = vis
        state.names = names
        st.image(vis, caption="Detecções por ícones", use_column_width=True)
        st.write(f"Total detecções: {len(icon_dets)}")
        if icon_dets:
            st.dataframe(
                [{"label":d["label"], "conf":round(d["conf"],3), "bbox":[int(x) for x in d["bbox"]]} for d in icon_dets],
                use_container_width=True
            )

# --- TAB 3: Validação Textos (OCR) ---
with tabs[2]:
    st.markdown("### Textos (OCR) com limpeza e deduplicação")
    if state.orig_img is None:
        st.info("Envie uma imagem na aba anterior.")
    else:
        with st.spinner("Executando OCR..."):
            text_boxes, vis_texts = ocr_text(
                state.orig_img,
                lang=cfg["ocr"]["lang"],
                min_conf=cfg["ocr"]["min_conf"],
                min_len=cfg["ocr"]["min_len"],
                stopwords=cfg["ocr"]["stopwords"],
                dedupe_sim=cfg["ocr"]["dedupe_sim"],
                join_nearby_px=cfg["ocr"]["join_nearby_px"]
            )
        state.text_boxes = text_boxes
        state.vis_texts = vis_texts
        st.image(vis_texts, caption="Textos detectados (OCR)", use_column_width=True)
        st.write(f"Total textos (únicos): {len(text_boxes)}")
        if text_boxes:
            editable = st.data_editor(
                [{"text":b["text"], "bbox":[int(x) for x in b["bbox"]]} for b in text_boxes],
                use_container_width=True, num_rows="dynamic"
            )
            # permite o usuário refinar/remover linhas
            if st.button("Aplicar edição"):
                # re-sincroniza text_boxes
                state.text_boxes = [{"text":row["text"], "bbox":row["bbox"]} for _,row in editable.iterrows()]
                st.success("Textos atualizados.")

# --- TAB 4: Análise STRIDE (Ollama) ---
with tabs[3]:
    st.markdown("### Núcleo STRIDE (combinando Ícones + Textos)")
    if state.orig_img is None or state.icon_dets is None or state.text_boxes is None:
        st.info("Faça a validação de ícones e textos nas abas anteriores.")
    else:
        if st.button("Executar análise STRIDE (Ollama)"):
            with st.spinner("Chamando LLM via Ollama..."):
                W,H = state.orig_img.size
                res = analyze_stride(state.icon_dets, state.text_boxes, cfg, (W,H))
            state.stride_result = res
            st.success("Análise concluída.")
        if state.stride_result:
            st.subheader("Resumo")
            js = state.stride_result.get("json")
            if js:
                st.write(js.get("stride_summary","(sem resumo)"))
                st.write(f"**Risco geral:** {js.get('overall_risk','?')}")
                st.write("**Top Mitigações:**", js.get("top_mitigations",[]))
                st.divider()
                st.subheader("Ameaças por categoria")
                for cat in ["spoofing","tampering","repudiation","information_disclosure","denial_of_service","elevation_of_privilege"]:
                    st.markdown(f"**{cat.replace('_',' ').title()}**")
                    items = js.get("threats",{}).get(cat,[])
                    if not items:
                        st.caption("— sem achados —")
                        continue
                    st.dataframe(items, use_container_width=True)
            else:
                st.warning("LLM não retornou JSON válido. Conteúdo bruto abaixo:")
                st.code(state.stride_result.get("raw",""), language="json")
            with st.expander("Prompt enviado ao LLM"):
                st.code(state.stride_result.get("prompt",""), language="markdown")

# --- TAB 5: Relatório (PDF) ---
with tabs[4]:
    st.markdown("### Exportar PDF")
    if not all([state.orig_img, state.vis_icons, state.vis_texts, state.stride_result]):
        st.info("Gere a análise primeiro.")
    else:
        if st.button("Gerar PDF"):
            with st.spinner("Montando PDF..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    pdf_path = tmp.name
                make_pdf_report(pdf_path, state.orig_img, state.vis_icons, state.vis_texts, state.stride_result, cfg)
                with open(pdf_path,"rb") as f:
                    st.download_button("Baixar Relatório PDF", data=f.read(), file_name="relatorio_stride.pdf", mime="application/pdf")
                st.success("PDF pronto!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, glob, json, random, re
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET
from PIL import Image
import yaml
import difflib
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ---------- util ----------

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def load_names(path):
    if path.lower().endswith((".yml", ".yaml")):
        with open(path, "r") as f: data = yaml.safe_load(f)
        names = data["names"]
    else:
        with open(path, "r") as f: names = [ln.strip() for ln in f if ln.strip()]
    return [str(x).strip() for x in names]

def load_class_map(path):
    if not path: return {}
    with open(path, "r") as f: return json.load(f)

def canon_label(raw, class_map, canonical_set):
    if raw is None: return None
    s = str(raw).strip()
    # 1) map explícito
    if s in class_map: s = class_map[s]
    # 2) normaliza simples
    s2 = re.sub(r"\s+", "_", s.strip().lower())
    if s in canonical_set: return s
    if s2 in canonical_set: return s2
    # 3) tenta map case-insensitive
    low_map = {k.lower(): v for k, v in class_map.items()}
    if s.lower() in low_map and low_map[s.lower()] in canonical_set:
        return low_map[s.lower()]
    return None

def find_image_for_xml(xml_path):
    """Resolve a imagem por <path>, <filename> ou busca por basename.* em pastas vizinhas."""
    try:
        root = ET.parse(xml_path).getroot()
        path_tag = root.findtext("path")
        fname_tag = root.findtext("filename")
    except Exception:
        return None

    xml_dir = os.path.dirname(xml_path)
    # 1) path do XML (absoluto ou relativo ao XML)
    if path_tag:
        cand = Path(path_tag)
        if not cand.is_absolute(): cand = Path(xml_dir) / path_tag
        if cand.is_file(): return str(cand.resolve())

    # 2) filename
    base = os.path.splitext(os.path.basename(fname_tag if fname_tag else xml_path))[0]

    # tenta no mesmo dir
    for e in IMG_EXTS:
        p = Path(xml_dir) / f"{base}{e}"
        if p.is_file(): return str(p.resolve())

    # tenta em subpastas comuns
    for sub in ("images", "image", "img", "JPEGImages", "imgs"):
        pdir = Path(xml_dir) / sub
        for e in IMG_EXTS:
            p = pdir / f"{base}{e}"
            if p.is_file(): return str(p.resolve())

    # tenta um nível acima
    up = Path(xml_dir).parent
    for e in IMG_EXTS:
        p = up / f"{base}{e}"
        if p.is_file(): return str(p.resolve())

    return None

def parse_boxes(root):
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        if bb is None: continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            out.append((name, [x1, y1, x2, y2]))
        except Exception:
            pass
    return out

def clip_box(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    return [x1, y1, x2, y2]

def to_yolo(xyxy, W, H, cls_id):
    x1, y1, x2, y2 = xyxy
    w = x2-x1; h = y2-y1
    if W<=0 or H<=0 or w<=0 or h<=0: return None
    xc = (x1+x2)/2.0/W; yc = (y1+y2)/2.0/H
    bw = w/W; bh = h/H
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

def write_txt(path, lines):
    with open(path, "w") as f: f.write("\n".join(lines))

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_root", required=True, help="raiz que contém XMLs (busca recursiva)")
    ap.add_argument("--out_root", required=True, help="saída YOLO (labels/, splits/, data.yaml)")
    ap.add_argument("--classes", required=True, help="classes.yaml/txt (vocabulário canônico, dos sintéticos)")
    ap.add_argument("--class_map", default=None, help="class_map.json (opcional)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--min_box_px", type=int, default=2)
    ap.add_argument("--copy_images", action="store_true")
    ap.add_argument("--auto_fuzzy", action="store_true", help="tentar casar rótulos próximos automaticamente")
    ap.add_argument("--fuzzy_cutoff", type=float, default=0.92, help="limiar do difflib.get_close_matches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    # vocabulário canônico
    names = load_names(args.classes)
    name_to_id = {n: i for i, n in enumerate(names)}
    canonical = set(names)
    cmap = load_class_map(args.class_map)

    # saída
    labels_dir = os.path.join(args.out_root, "labels"); ensure_dir(labels_dir)
    images_dir = os.path.join(args.out_root, "images") if args.copy_images else None
    if images_dir: ensure_dir(images_dir)
    splits_dir = os.path.join(args.out_root, "splits"); ensure_dir(splits_dir)

    # coleta xmls
    xmls = sorted(glob.glob(os.path.join(args.xml_root, "**", "*.xml"), recursive=True))
    if not xmls:
        raise FileNotFoundError(f"Nenhum XML em {args.xml_root}")

    kept_images = []
    stats = Counter()
    per_class = Counter()
    skipped = Counter()
    unknowns = Counter()
    suggestions = defaultdict(Counter)

    for xml in xmls:
        img_path = find_image_for_xml(xml)
        if not img_path or not os.path.isfile(img_path):
            skipped["img_not_found"] += 1
            continue

        # tamanho real da imagem
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            skipped["img_open_fail"] += 1
            continue

        try:
            root = ET.parse(xml).getroot()
        except Exception:
            skipped["xml_parse_fail"] += 1
            continue

        yolo_lines = []
        for raw_label, xyxy in parse_boxes(root):
            lab = canon_label(raw_label, cmap, canonical)
            if lab is None:
                # tentar sugestão fuzzy
                raw_norm = re.sub(r"\s+", "_", str(raw_label).strip().lower())
                close = difflib.get_close_matches(raw_norm, names, n=1, cutoff=args.fuzzy_cutoff)
                if args.auto_fuzzy and close:
                    lab = close[0]
                else:
                    unknowns[raw_label] += 1
                    if close:
                        suggestions[raw_label][close[0]] += 1
                    continue

            cls_id = name_to_id[lab]
            x1,y1,x2,y2 = clip_box(xyxy, W, H)
            if (x2-x1) < args.min_box_px or (y2-y1) < args.min_box_px:
                skipped["tiny_box"] += 1
                continue
            line = to_yolo([x1,y1,x2,y2], W, H, cls_id)
            if line:
                yolo_lines.append(line)
                per_class[lab] += 1
                stats["boxes"] += 1

        if yolo_lines:
            base = os.path.splitext(os.path.basename(img_path))[0]
            write_txt(os.path.join(labels_dir, base + ".txt"), yolo_lines)
            if images_dir:
                from shutil import copy2
                dst = os.path.join(images_dir, os.path.basename(img_path))
                if os.path.abspath(dst) != os.path.abspath(img_path):
                    copy2(img_path, dst)
                kept_images.append(os.path.abspath(dst))
            else:
                kept_images.append(os.path.abspath(img_path))
            stats["images"] += 1
        else:
            skipped["no_labels_after_filter"] += 1

    # splits
    random.shuffle(kept_images)
    n_val = max(1, int(len(kept_images)*args.val_ratio))
    val_imgs = kept_images[:n_val]
    trn_imgs = kept_images[n_val:]
    write_txt(os.path.join(splits_dir, "train.txt"), trn_imgs)
    write_txt(os.path.join(splits_dir, "val.txt"), val_imgs)

    # data.yaml (aponta para os splits)
    data_yaml = {
        "train": [os.path.join(splits_dir, "train.txt")],
        "val":   [os.path.join(splits_dir, "val.txt")],
        "nc": len(names),
        "names": names
    }
    with open(os.path.join(args.out_root, "data.yaml"), "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    # relatórios
    report_dir = os.path.join(args.out_root, "reports"); ensure_dir(report_dir)
    # unknowns e sugestões
    if unknowns:
        with open(os.path.join(report_dir, "unknown_labels.tsv"), "w") as f:
            f.write("raw_label\tcount\tsuggestion\n")
            for lbl, cnt in unknowns.most_common():
                sug = ",".join([f"{k}({v})" for k,v in suggestions[lbl].most_common()]) or "-"
                f.write(f"{lbl}\t{cnt}\t{sug}\n")

    with open(os.path.join(report_dir, "class_counts.tsv"), "w") as f:
        for k,v in sorted(per_class.items()):
            f.write(f"{k}\t{v}\n")

    print("\n[OK] VOC→YOLO finalizado")
    print("  imagens geradas :", stats['images'])
    print("  caixas geradas  :", stats['boxes'])
    if unknowns:
        print("  ⚠ rótulos desconhecidos:", sum(unknowns.values()), "→ veja", os.path.join(report_dir, "unknown_labels.tsv"))
    if skipped:
        print("  avisos (descartes):", dict(skipped))
    print("  data.yaml:", os.path.join(args.out_root, "data.yaml"))
    print("  splits   :", os.path.join(splits_dir, "train.txt"), "e", os.path.join(splits_dir, "val.txt"))

if __name__ == "__main__":
    main()

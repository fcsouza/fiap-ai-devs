#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, glob, json, shutil
from pathlib import Path
import argparse
import difflib
import yaml

# ------------------------
# Helpers
# ------------------------

def clean_name(x: str) -> str:
    x = x.strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"__+", "_", x)
    return x

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def read_lines(p):
    with open(p, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def write_lines(p, lines):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(lines))

def symlink_dir(src, dst):
    dst = Path(dst)
    if dst.exists():
        if dst.is_symlink() or dst.is_dir():
            return
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except FileExistsError:
        pass

# ------------------------
# Synonyms (real → canonical)
# Chaveada em nomes normalizados
# ------------------------

SYNONYMS = {
    # ===== AWS =====
    "aws_cloudfront": "aws_amazon_cloudfront",
    "aws_cloudwatch": "aws_amazon_cloudwatch",
    "aws_dynamodb_table": "aws_amazon_dynamodb",
    "aws_ec2_instance": "aws_amazon_ec2",
    "aws_ec2_instances": "aws_amazon_ec2",
    "aws_virtual_private_cloud": "aws_amazon_virtual_private_cloud",
    "aws_vpc_virtual_private_cloud_vpc": "aws_amazon_virtual_private_cloud",
    "aws_elastic_block_store_volume": "aws_amazon_elastic_block_store",
    "aws_elasticache": "aws_amazon_elasticache",
    "aws_rds": "aws_amazon_rds",
    "aws_route_53_hosted_zone": "aws_amazon_route_53",
    "aws_simple_notification_service_topic": "aws_amazon_simple_notification_service",
    "aws_simple_queue_service_queue": "aws_amazon_simple_queue_service",
    "aws_simple_storage_service_bucket": "aws_amazon_simple_storage_service",
    "aws_simple_storage_service_bucket_with_objects": "aws_amazon_simple_storage_service",
    "aws_simple_storage_service_object": "aws_amazon_simple_storage_service",
    "aws_simple_storage_service_s3_standard": "aws_amazon_simple_storage_service",
    "aws_identity_access_management_role": "aws_identity_and_access_management",
    "aws_auto_scaling": "aws_amazon_ec2_auto_scaling",
    "aws_autoscaling": "aws_amazon_ec2_auto_scaling",
    "aws_elastic_container_service_service": "aws_amazon_elastic_container_service",
    "aws_elastic_container_service_container_2": "aws_amazon_elastic_container_service",
    "aws_elastic_load_balancing_application_load_balancer": "aws_application_load_balancer",
    "aws_elastic_load_balancing_network_load_balancer": "aws_elastic_load_balancing",
    "aws_elactic_file_systemnfs_multi_az": "aws_elastic_file_system",  # typo do real
    "aws_lambda_lambda_function": "aws_lambda",
    "aws_cloudformation_template": "aws_cloudformation",

    # genéricos/sem classe canônica (drop = None)
    "aws_cloud": None,
    "aws_region": None,

    # ===== Azure =====
    "azure_sql": "azure_sql_database",
    "azure_machine_learning_studio_workspaces": "azure_machine_learning",
    "logic_apps": "azure_logic_apps",

    # genéricos (drop)
    "azure_services": None,
    "azure_resource_groups": None,
    "resource_group": None,
    "developer_portal": None,

    # ===== Outros sem canônico nos sintéticos (drop) =====
    "sass_services": None,
    "sei_sip": None,
    "solr": None,
    "user": None,
}

# ------------------------
# Core
# ------------------------

def build_mapping(real_names, canon_names, synonyms=None, fallback="drop", cutoff=0.6):
    """
    Retorna:
      - real_to_canon_name: dict nome_real -> nome_canon (ou None se drop)
      - real_id_to_canon_id: dict id_real -> id_canon (somente mapeados)
      - report: dict com estatísticas
    """
    synonyms = synonyms or {}
    canon_set = set(canon_names)
    canon_norm = [clean_name(n) for n in canon_names]
    canon_by_norm = {clean_name(n): n for n in canon_names}

    real_to_canon_name = {}
    unmapped = []
    used_via_exact = []
    used_via_syn = []
    used_via_fuzzy = []

    # pass 1: exact
    for n in real_names:
        n_clean = clean_name(n)
        if n in canon_names:
            real_to_canon_name[n] = n
            used_via_exact.append(n)
        elif n_clean in canon_by_norm:
            real_to_canon_name[n] = canon_by_norm[n_clean]
            used_via_exact.append(n)
        else:
            real_to_canon_name[n] = None  # placeholder
            unmapped.append(n)

    # pass 2: synonyms
    still_unmapped = [n for n in real_names if real_to_canon_name[n] is None]
    for n in still_unmapped:
        key = clean_name(n)
        if key in synonyms:
            tgt = synonyms[key]
            if tgt is None:
                real_to_canon_name[n] = None  # drop explícito
            elif tgt in canon_names or clean_name(tgt) in canon_by_norm:
                real_to_canon_name[n] = canon_by_norm.get(clean_name(tgt), tgt)
                used_via_syn.append(n)
            else:
                real_to_canon_name[n] = None
        # else continua None para fallback

    # pass 3: fallback fuzzy
    if fallback == "nearest":
        still_unmapped = [n for n in real_names if real_to_canon_name[n] is None]
        for n in still_unmapped:
            # heurística: remover sufixos comuns antes do fuzzy
            base = re.sub(r"_(bucket|service|services|topic|queue|object|objects|table|instance|instances|hosted_zone)$", "", clean_name(n))
            cand = difflib.get_close_matches(base, canon_norm, n=1, cutoff=cutoff)
            if cand:
                real_to_canon_name[n] = canon_by_norm[cand[0]]
                used_via_fuzzy.append(n)
            else:
                real_to_canon_name[n] = None

    # id maps
    canon_id = {name: i for i, name in enumerate(canon_names)}
    real_id_to_canon_id = {}
    dropped = []
    for i, n in enumerate(real_names):
        tgt = real_to_canon_name[n]
        if tgt is None:
            dropped.append(n)
        else:
            real_id_to_canon_id[i] = canon_id[tgt]

    report = {
        "total_real_classes": len(real_names),
        "total_canon_classes": len(canon_names),
        "mapped_exact": len(used_via_exact),
        "mapped_synonyms": len(used_via_syn),
        "mapped_fuzzy": len(used_via_fuzzy),
        "dropped": len(dropped),
        "dropped_names": sorted(set(dropped)),
    }
    return real_to_canon_name, real_id_to_canon_id, report

def remap_label_file(src_txt, dst_txt, id_map):
    lines_in = []
    try:
        with open(src_txt, "r") as f:
            lines_in = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        # imagem sem label -> cria vazio
        write_lines(dst_txt, [])
        return 0, 0

    kept = []
    dropped = 0
    for ln in lines_in:
        parts = ln.split()
        if len(parts) != 5:
            dropped += 1
            continue
        try:
            cid = int(float(parts[0]))
            rest = parts[1:]
        except Exception:
            dropped += 1
            continue
        if cid not in id_map:
            dropped += 1
            continue
        new_cid = id_map[cid]
        kept.append(" ".join([str(new_cid)] + rest))

    write_lines(dst_txt, kept)
    return len(kept), dropped

def convert_dataset(real_yaml, canon_yaml, out_dir, link_images=True, fallback="drop", cutoff=0.6, synonyms_json=None):
    real_cfg = load_yaml(real_yaml)
    canon_cfg = load_yaml(canon_yaml)

    canon_names = canon_cfg["names"]
    real_names = real_cfg.get("names") or []
    if not real_names:
        raise RuntimeError("YAML dos dados reais não contém 'names'.")

    # resolve caminhos
    base = Path(real_cfg.get("path", Path(real_yaml).parent)).resolve()
    train_imgs = Path(base, real_cfg["train"]).resolve()
    val_imgs   = Path(base, real_cfg["val"]).resolve()
    train_lbls = Path(str(train_imgs).replace("/images/", "/labels/"))
    val_lbls   = Path(str(val_imgs).replace("/images/", "/labels/"))

    # synonyms externos
    synonyms = dict(SYNONYMS)
    if synonyms_json:
        with open(synonyms_json, "r") as f:
            ext = json.load(f)
        # normaliza chaves/valores
        for k, v in ext.items():
            k2 = clean_name(k)
            v2 = None if v is None else clean_name(v)
            synonyms[k2] = v if v is None else v  # manter valor "humano"
    
    # mapping
    real_to_canon_name, id_map, report = build_mapping(
        real_names, canon_names, synonyms=synonyms, fallback=fallback, cutoff=cutoff
    )

    print("[MAPA] real → canônico:")
    for rn in real_names:
        print(f"  {rn}  →  {real_to_canon_name[rn]}")
    print("\n[RESUMO]", json.dumps(report, ensure_ascii=False, indent=2))

    # preparar out_dir
    out_dir = Path(out_dir).resolve()
    out_imgs_train = out_dir / "images" / "train"
    out_imgs_val   = out_dir / "images" / "val"
    out_lbls_train = out_dir / "labels" / "train"
    out_lbls_val   = out_dir / "labels" / "val"
    out_lbls_train.mkdir(parents=True, exist_ok=True)
    out_lbls_val.mkdir(parents=True, exist_ok=True)

    # link/copiar imagens
    if link_images:
        symlink_dir(str(train_imgs), str(out_imgs_train))
        symlink_dir(str(val_imgs), str(out_imgs_val))
    else:
        # copia (lento e consome espaço)
        for src_dir, dst_dir in [(train_imgs, out_imgs_train), (val_imgs, out_imgs_val)]:
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            for p in glob.glob(str(Path(src_dir, "*"))):
                q = Path(dst_dir, Path(p).name)
                if not q.exists():
                    shutil.copy2(p, q)

    # remap labels
    def remap_split(src_imgs_dir, src_lbls_dir, dst_lbls_dir):
        img_exts = (".jpg", ".jpeg", ".png")
        n_img = 0
        n_box_kept = 0
        n_box_drop = 0
        for img_path in glob.glob(str(Path(src_imgs_dir, "*"))):
            if not img_path.lower().endswith(img_exts):
                continue
            n_img += 1
            base = Path(img_path).stem
            src_txt = Path(src_lbls_dir, base + ".txt")
            dst_txt = Path(dst_lbls_dir, base + ".txt")
            kept, dropped = remap_label_file(src_txt, dst_txt, id_map)
            n_box_kept += kept
            n_box_drop += dropped
        return n_img, n_box_kept, n_box_drop

    print("\n[REMAP] train…")
    t_imgs, t_kept, t_drop = remap_split(train_imgs, train_lbls, out_lbls_train)
    print(f"train: imagens={t_imgs}  caixas_kept={t_kept}  caixas_drop={t_drop}")

    print("[REMAP] val…")
    v_imgs, v_kept, v_drop = remap_split(val_imgs, val_lbls, out_lbls_val)
    print(f"val:   imagens={v_imgs}  caixas_kept={v_kept}  caixas_drop={v_drop}")

    # novo YAML canônico apontando para out_dir
    out_yaml = out_dir / "dataset_synced.yaml"
    out_cfg = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": canon_names
    }
    save_yaml(out_cfg, out_yaml)
    print("\n[OK] Dataset convertido:")
    print(" - YAML:", out_yaml)
    print(" - Imagens:", out_dir / "images")
    print(" - Labels :", out_dir / "labels")

def main():
    ap = argparse.ArgumentParser(description="Remapear labels YOLO de um dataset real para classes canônicas dos sintéticos.")
    ap.add_argument("--real-yaml", required=True, help="YAML do dataset real (com 'names', 'path', 'train', 'val').")
    ap.add_argument("--canon-yaml", required=True, help="YAML canônico (dos sintéticos, com 'names').")
    ap.add_argument("--out-dataset", required=True, help="Pasta de saída do dataset convertido.")
    ap.add_argument("--fallback", choices=["drop","nearest"], default="drop",
                    help="Como tratar classes reais sem mapeamento: 'drop' (descarta) ou 'nearest' (fuzzy).")
    ap.add_argument("--cutoff", type=float, default=0.6, help="Cutoff do fuzzy quando fallback=nearest.")
    ap.add_argument("--synonyms-json", type=str, default=None,
                    help="JSON opcional com mapeamentos extras real->canônico.")
    ap.add_argument("--no-link", action="store_true", help="Se setado, copia imagens em vez de symlink (mais lento).")
    args = ap.parse_args()

    convert_dataset(
        real_yaml=args.real_yaml,
        canon_yaml=args.canon_yaml,
        out_dir=args.out_dataset,
        link_images=not args.no_link,
        fallback=args.fallback,
        cutoff=args.cutoff,
        synonyms_json=args.synonyms_json
    )

if __name__ == "__main__":
    main()

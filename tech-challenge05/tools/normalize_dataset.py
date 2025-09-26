#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, random, argparse, difflib
from pathlib import Path
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET
from PIL import Image
import yaml
from tqdm.auto import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ---------------------- util ----------------------
def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)
def read_yaml(p): return yaml.safe_load(open(p, "r"))
def write_yaml(obj, p): ensure_dir(Path(p).parent); yaml.safe_dump(obj, open(p, "w"), sort_keys=False, allow_unicode=True)

def write_txt(p, lines): ensure_dir(Path(p).parent); open(p, "w").write("\n".join(lines))
def clean_name(x:str)->str:
    x = x.strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"__+", "_", x)

def load_names(path):
    if path.lower().endswith((".yml",".yaml")):
        return [str(x).strip() for x in read_yaml(path)["names"]]
    return [ln.strip() for ln in open(path) if ln.strip()]

# ------------------ sinônimos base ----------------
SYNONYMS_DEFAULT = {
    # ===== AWS =====
    "aws_cloudfront":"aws_amazon_cloudfront",
    "aws_cloudwatch":"aws_amazon_cloudwatch",
    "aws_dynamodb_table":"aws_amazon_dynamodb",
    "aws_ec2_instance":"aws_amazon_ec2",
    "aws_ec2_instances":"aws_amazon_ec2",
    "aws_virtual_private_cloud":"aws_amazon_virtual_private_cloud",
    "aws_vpc_virtual_private_cloud_vpc":"aws_amazon_virtual_private_cloud",
    "aws_elastic_block_store_volume":"aws_amazon_elastic_block_store",
    "aws_elasticache":"aws_amazon_elasticache",
    "aws_rds":"aws_amazon_rds",
    "aws_route_53_hosted_zone":"aws_amazon_route_53",
    "aws_simple_notification_service_topic":"aws_amazon_simple_notification_service",
    "aws_simple_queue_service_queue":"aws_amazon_simple_queue_service",
    "aws_simple_storage_service_bucket":"aws_amazon_simple_storage_service",
    "aws_simple_storage_service_bucket_with_objects":"aws_amazon_simple_storage_service",
    "aws_simple_storage_service_object":"aws_amazon_simple_storage_service",
    "aws_simple_storage_service_s3_standard":"aws_amazon_simple_storage_service",
    "aws_identity_access_management_role":"aws_identity_and_access_management",
    "aws_auto_scaling":"aws_amazon_ec2_auto_scaling",
    "aws_autoscaling":"aws_amazon_ec2_auto_scaling",
    "aws_elastic_container_service_service":"aws_amazon_elastic_container_service",
    "aws_elastic_container_service_container_2":"aws_amazon_elastic_container_service",
    "aws_elastic_load_balancing_application_load_balancer":"aws_application_load_balancer",
    "aws_elastic_load_balancing_network_load_balancer":"aws_elastic_load_balancing",
    "aws_elactic_file_systemnfs_multi_az":"aws_elastic_file_system",
    "aws_lambda_lambda_function":"aws_lambda",
    "aws_cloudformation_template":"aws_cloudformation",
    # “genéricos” que vamos descartar
    "aws_cloud": None, "aws_region": None,

    # ===== Azure =====
    "azure_sql":"azure_sql_database",
    "azure_machine_learning_studio_workspaces":"azure_machine_learning",
    "logic_apps":"azure_logic_apps",
    # “genéricos” (drop)
    "azure_services": None, "azure_resource_groups": None, "resource_group": None,
    "developer_portal": None,

    # ===== Outros (sem classe nos sintéticos) =====
    "sass_services": None, "sei_sip": None, "solr": None, "user": None,
}

def build_mapper(real_names, canon_names, synonyms=None, fallback="drop", cutoff=0.92):
    synonyms = {**SYNONYMS_DEFAULT, **(synonyms or {})}
    canon_norm = {clean_name(n): n for n in canon_names}
    name_to_id = {n:i for i,n in enumerate(canon_names)}

    real2canon = {}
    mapped_exact, mapped_syn, mapped_fuzzy, dropped = [],[],[],[]

    # exact / normalized
    for rn in real_names:
        if rn in canon_names:
            real2canon[rn] = rn; mapped_exact.append(rn); continue
        rn_norm = clean_name(rn)
        if rn_norm in canon_norm:
            real2canon[rn] = canon_norm[rn_norm]; mapped_exact.append(rn); continue
        real2canon[rn] = None

    # synonyms
    for rn in real_names:
        if real2canon[rn] is not None: continue
        key = clean_name(rn)
        if key in synonyms:
            tgt = synonyms[key]
            if tgt is None:
                real2canon[rn] = None; dropped.append(rn)
            else:
                real2canon[rn] = canon_norm.get(clean_name(tgt), tgt)
                mapped_syn.append(rn)

    # fuzzy (opcional)
    if fallback == "nearest":
        canon_list_norm = list(canon_norm.keys())
        for rn in real_names:
            if real2canon[rn] is not None: continue
            base = re.sub(r"_(bucket|service|services|topic|queue|object|objects|table|instance|instances|hosted_zone)$",
                          "", clean_name(rn))
            match = difflib.get_close_matches(base, canon_list_norm, n=1, cutoff=cutoff)
            if match:
                real2canon[rn] = canon_norm[match[0]]; mapped_fuzzy.append(rn)
            else:
                dropped.append(rn)

    id_map = {i: name_to_id[real2canon[rn]] for i, rn in enumerate(real_names) if real2canon[rn] is not None}
    report = {
        "mapped_exact": len(mapped_exact), "mapped_synonyms": len(mapped_syn),
        "mapped_fuzzy": len(mapped_fuzzy), "dropped": len(dropped),
        "dropped_names": sorted(set(dropped)),
    }
    return real2canon, id_map, report

# ----------------- imagem para XML ----------------
def find_image_for_xml(xml_path, img_roots):
    root = ET.parse(xml_path).getroot()
    xml_dir = Path(xml_path).parent

    # 1) path do XML
    path_tag = root.findtext("path")
    if path_tag:
        p = Path(path_tag)
        if not p.is_absolute(): p = xml_dir / p
        if p.is_file(): return str(p.resolve())

    # 2) filename
    base = Path(root.findtext("filename") or Path(xml_path).stem).stem

    # 3) mesmo diretório
    for e in IMG_EXTS:
        p = xml_dir / f"{base}{e}"
        if p.is_file(): return str(p.resolve())

    # 4) procurar em img_roots (recursivo, por basename)
    for r in img_roots:
        for p in Path(r).rglob(f"{base}*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                return str(p.resolve())

    # 5) subpastas comuns ao lado do XML
    for sub in ("images","image","img","JPEGImages","imgs"):
        for e in IMG_EXTS:
            p = xml_dir/sub/f"{base}{e}"
            if p.is_file(): return str(p.resolve())

    return None

# ----------------- VOC → YOLO ---------------------
def voc_to_yolo(xml_root, out_root, classes_yaml, val_ratio=0.1, img_roots=None,
                min_box_px=2, synonyms=None, fallback="drop", cutoff=0.92,
                copy_images=False, seed=0, auto_fuzzy=False):
    random.seed(seed)
    names = load_names(classes_yaml); name_to_id = {n:i for i,n in enumerate(names)}
    img_roots = img_roots or []
    labels_dir = Path(out_root, "labels"); images_dir = Path(out_root, "images") if copy_images else None
    splits_dir = Path(out_root, "splits")
    ensure_dir(labels_dir); ensure_dir(splits_dir); 
    if images_dir: ensure_dir(images_dir)

    # mapeamento de rótulos
    real_names_from_xml = set()
    for x in glob.glob(str(Path(xml_root, "**/*.xml")), recursive=True):
        try:
            for obj in ET.parse(x).getroot().findall("object"):
                nm = obj.findtext("name")
                if nm: real_names_from_xml.add(nm)
        except Exception: pass

    real2canon, _, rep = build_mapper(sorted(real_names_from_xml), names, synonyms=synonyms,
                                      fallback=("nearest" if auto_fuzzy else fallback), cutoff=cutoff)
    unknown = {rn for rn in real_names_from_xml if real2canon.get(rn) is None}

    kept_images, per_class = [], Counter()
    skipped = Counter()
    xml_files = sorted(glob.glob(str(Path(xml_root, "**/*.xml")), recursive=True))
    for xml in tqdm(xml_files, desc="VOC→YOLO", unit="xml"):
        img = find_image_for_xml(xml, img_roots)
        if not img: skipped["img_not_found"] += 1; continue

        try:
            with Image.open(img) as im: W,H = im.size
        except Exception: skipped["img_open_fail"] += 1; continue

        root = ET.parse(xml).getroot()
        yolo_lines=[]
        for obj in root.findall("object"):
            raw = obj.findtext("name")
            canon = real2canon.get(raw)
            if canon is None: skipped["unknown_label"] += 1; continue
            cls = name_to_id[canon]
            bb = obj.find("bndbox"); 
            if bb is None: continue
            try:
                x1=float(bb.findtext("xmin")); y1=float(bb.findtext("ymin"))
                x2=float(bb.findtext("xmax")); y2=float(bb.findtext("ymax"))
            except: continue
            x1,x2 = sorted([max(0,min(W-1,x1)), max(0,min(W-1,x2))])
            y1,y2 = sorted([max(0,min(H-1,y1)), max(0,min(H-1,y2))])
            if (x2-x1)<min_box_px or (y2-y1)<min_box_px: skipped["tiny_box"]+=1; continue
            xc=(x1+x2)/2/W; yc=(y1+y2)/2/H; bw=(x2-x1)/W; bh=(y2-y1)/H
            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            per_class[canon]+=1

        if not yolo_lines: skipped["no_labels_after_filter"]+=1; continue

        base = Path(img).stem
        write_txt(Path(labels_dir, base+".txt"), yolo_lines)
        if images_dir:
            from shutil import copy2
            dst = Path(images_dir, Path(img).name)
            if str(dst.resolve()) != str(Path(img).resolve()): copy2(img, dst)
            kept_images.append(str(dst.resolve()))
        else:
            kept_images.append(str(Path(img).resolve()))

    # splits + data.yaml
    random.shuffle(kept_images)
    n_val = max(1, int(len(kept_images)*val_ratio))
    write_txt(Path(splits_dir,"train.txt"), kept_images[n_val:])
    write_txt(Path(splits_dir,"val.txt"),   kept_images[:n_val])
    write_yaml({"train":[str(Path(splits_dir,'train.txt'))],
                "val":[str(Path(splits_dir,'val.txt'))],
                "nc":len(names),"names":names},
               Path(out_root,"data.yaml"))

    # reports
    rep_dir = Path(out_root, "reports"); ensure_dir(rep_dir)
    if unknown:
        write_txt(Path(rep_dir,"unknown_labels.txt"), sorted(unknown))
    write_txt(Path(rep_dir,"class_counts.tsv"), [f"{k}\t{v}" for k,v in sorted(per_class.items())])

    print("\n[OK] VOC→YOLO")
    print("  imagens geradas:", len(kept_images))
    print("  caixas geradas :", sum(per_class.values()))
    print("  skipped        :", dict(skipped))
    print("  data.yaml      :", Path(out_root,"data.yaml"))
    if unknown:
        print("  ⚠ rótulos desconhecidos →", Path(rep_dir,"unknown_labels.txt"))
    print("  resumo mapa    :", rep)

# --------------- YOLO → YOLO (remap) ---------------
def remap_yolo(real_yaml, canon_yaml, out_dataset, synonyms=None, fallback="drop", cutoff=0.92, link_images=True):
    real_cfg = read_yaml(real_yaml); canon_cfg = read_yaml(canon_yaml)
    canon_names = canon_cfg["names"]; real_names = real_cfg["names"]

    # localizar imagens/labels do dataset real
    base = Path(real_cfg.get("path", Path(real_yaml).parent)).resolve()
    train_imgs = Path(base, real_cfg["train"]).resolve()
    val_imgs   = Path(base, real_cfg["val"]).resolve()
    train_lbls = Path(str(train_imgs).replace("/images/","/labels/"))
    val_lbls   = Path(str(val_imgs).replace("/images/","/labels/"))

    real2canon, id_map, rep = build_mapper(real_names, canon_names, synonyms=synonyms, fallback=fallback, cutoff=cutoff)
    print("[MAPA] real→canônico:", json.dumps(real2canon, ensure_ascii=False, indent=2))
    print("[RESUMO]", json.dumps(rep, ensure_ascii=False, indent=2))

    out_dir = Path(out_dataset); 
    out_imgs_tr = out_dir/"images/train"; out_imgs_va = out_dir/"images/val"
    out_lbls_tr = out_dir/"labels/train"; out_lbls_va = out_dir/"labels/val"
    for d in (out_imgs_tr,out_imgs_va,out_lbls_tr,out_lbls_va): ensure_dir(d)

    # link/copiar imagens
    if link_images:
        for src, dst in [(train_imgs,out_imgs_tr),(val_imgs,out_imgs_va)]:
            if not Path(dst).exists():
                os.symlink(src, dst, target_is_directory=True)
    else:
        from shutil import copy2
        for src, dst in [(train_imgs,out_imgs_tr),(val_imgs,out_imgs_va)]:
            for p in Path(src).glob("*"):
                q = Path(dst, p.name)
                if not q.exists(): copy2(p, q)

    def remap_split(src_imgs, src_lbls, dst_lbls):
        kept = drop = imgc = 0
        for img in tqdm(Path(src_imgs).glob("*"), desc=f"Remap images", unit="img"):
            if img.suffix.lower() not in IMG_EXTS: continue
            imgc += 1
            txt_in = Path(src_lbls, img.stem + ".txt")
            lines = []
            if txt_in.exists():
                for ln in open(txt_in):
                    ps = ln.strip().split()
                    if len(ps)!=5: drop+=1; continue
                    try: cid = int(float(ps[0])); 
                    except: drop+=1; continue
                    if cid not in id_map: drop+=1; continue
                    ps[0] = str(id_map[cid]); lines.append(" ".join(ps)); kept+=1
            write_txt(Path(dst_lbls, img.stem+".txt"), lines)
        return imgc, kept, drop

    t_imgs, t_kept, t_drop = remap_split(train_imgs, train_lbls, out_lbls_tr)
    v_imgs, v_kept, v_drop = remap_split(val_imgs,   val_lbls,   out_lbls_va)
    print(f"[REMAP] train imgs={t_imgs} kept={t_kept} drop={t_drop}")
    print(f"[REMAP] val   imgs={v_imgs} kept={v_kept} drop={v_drop}")

    write_yaml({"path": str(out_dir.resolve()), "train":"images/train", "val":"images/val", "names": canon_names},
               out_dir/"dataset_synced.yaml")
    print("[OK] YOLO→YOLO concluído:", out_dir/"dataset_synced.yaml")

# ------------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="Normalizar dataset para classes canônicas (sintéticos).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("from_voc", help="Converter VOC XML + imagens → YOLO canônico")
    a.add_argument("--xml_root", required=True)
    a.add_argument("--out_root", required=True)
    a.add_argument("--classes", required=True)
    a.add_argument("--img_roots", nargs="*", default=[], help="pastas onde procurar imagens (recursivo)")
    a.add_argument("--val_ratio", type=float, default=0.1)
    a.add_argument("--min_box_px", type=int, default=2)
    a.add_argument("--class_map_json", default=None)
    a.add_argument("--fallback", choices=["drop","nearest"], default="drop")
    a.add_argument("--cutoff", type=float, default=0.92)
    a.add_argument("--copy_images", action="store_true")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--auto_fuzzy", action="store_true")

    b = sub.add_parser("remap_yolo", help="Remapear YOLO→YOLO para classes canônicas")
    b.add_argument("--real_yaml", required=True)
    b.add_argument("--canon_yaml", required=True)
    b.add_argument("--out_dataset", required=True)
    b.add_argument("--class_map_json", default=None)
    b.add_argument("--fallback", choices=["drop","nearest"], default="drop")
    b.add_argument("--cutoff", type=float, default=0.92)
    b.add_argument("--no_link", action="store_true")

    args = ap.parse_args()
    synonyms = {}
    if args.cmd == "from_voc":
        if args.class_map_json and Path(args.class_map_json).exists():
            synonyms = json.load(open(args.class_map_json))
        voc_to_yolo(args.xml_root, args.out_root, args.classes,
                    val_ratio=args.val_ratio, img_roots=args.img_roots,
                    min_box_px=args.min_box_px, synonyms=synonyms,
                    fallback=args.fallback, cutoff=args.cutoff,
                    copy_images=args.copy_images, seed=args.seed,
                    auto_fuzzy=args.auto_fuzzy)
    else:
        if args.class_map_json and Path(args.class_map_json).exists():
            synonyms = json.load(open(args.class_map_json))
        remap_yolo(args.real_yaml, args.canon_yaml, args.out_dataset,
                   synonyms=synonyms, fallback=args.fallback, cutoff=args.cutoff,
                   link_images=not args.no_link)

if __name__ == "__main__":
    main()

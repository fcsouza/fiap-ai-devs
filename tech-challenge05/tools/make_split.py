# tools/make_split.py
import os, glob, random, argparse, re
from pathlib import Path
from collections import Counter, defaultdict

def find_images(root, exts):
    imgs = []
    for e in exts:
        imgs += glob.glob(str(Path(root, "images", f"*{e}")))
    # tirar duplicados e normalizar
    imgs = sorted(set(str(Path(p)) for p in imgs))
    return imgs

def label_path_for(img_path, root, labels_dir="labels"):
    p = Path(img_path)
    # troca "images" -> "labels" e extensão por .txt
    return str(Path(root, labels_dir, p.stem + ".txt"))

def parse_classes_from_label(txt_path):
    """Lê a 1ª coluna de cada linha YOLO e retorna set de ints."""
    try:
        with open(txt_path, "r") as f:
            cls_ids = []
            for line in f:
                line=line.strip()
                if not line: 
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                    if cid >= 0:
                        cls_ids.append(cid)
                except Exception:
                    continue
            return set(cls_ids), len(cls_ids)
    except FileNotFoundError:
        return set(), 0

def write_list(paths, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(paths))

def print_hist(name, cls_hist, n_imgs, n_boxes):
    if not cls_hist:
        print(f"{name}: imgs={n_imgs} boxes={n_boxes} (sem classes)")
        return
    top = ", ".join(f"{k}:{v}" for k,v in sorted(cls_hist.items())[:10])
    print(f"{name}: imgs={n_imgs} boxes={n_boxes} classes={len(cls_hist)}  (ex: {top})")

def greedy_cover_validation(imgs, img_classes, target_count):
    """
    Garante ao menos 1 imagem por classe no 'val' (se possível),
    depois completa aleatoriamente até atingir target_count.
    """
    # frequência de cada classe no dataset inteiro
    cls_freq = Counter()
    for s in img_classes.values():
        cls_freq.update(s)

    # classes raras primeiro
    rare_to_common = [c for c,_ in sorted(cls_freq.items(), key=lambda kv: kv[1])]
    chosen = set()
    used = set()
    for c in rare_to_common:
        # escolhe a 1a imagem que contenha c e ainda não esteja no val
        for img in imgs:
            if img in used: 
                continue
            if c in img_classes[img]:
                chosen.add(img)
                used.add(img)
                break
        if len(chosen) >= target_count:
            return list(chosen)

    # completa aleatoriamente
    remaining = [p for p in imgs if p not in chosen]
    random.shuffle(remaining)
    for p in remaining:
        if len(chosen) >= target_count:
            break
        chosen.add(p)
    return list(chosen)

def group_by_prefix(imgs, prefix_len):
    groups = defaultdict(list)
    for p in imgs:
        key = Path(p).stem[:prefix_len]
        groups[key].append(p)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="pasta do dataset (tem 'images' e 'labels')")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--exts", default=".jpg,.jpeg,.png", help="extensões válidas separadas por vírgula")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels-dir", default="labels", help="nome da pasta de labels (default: labels)")
    ap.add_argument("--stratify", action="store_true",
                    help="tenta garantir cobertura de classes no conjunto de validação")
    ap.add_argument("--group-by-prefix", type=int, default=0,
                    help="evita vazamento: coloca grupos pelo prefixo do nome (N chars) todos no mesmo split")
    ap.add_argument("--drop-missing-labels", action="store_true",
                    help="descarta imagens sem .txt de label (por padrão mantém como background)")
    ap.add_argument("--yaml-out", type=str, default=None,
                    help="se definido, gera um YAML com train/val (lista de arquivos) e tenta ler names de classes.yaml")
    args = ap.parse_args()

    random.seed(args.seed)

    root = Path(args.root).resolve()
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    imgs = find_images(root, exts)
    if not imgs:
        print("[ERRO] Nenhuma imagem encontrada em", Path(root, "images"))
        return

    # coleta classes por imagem + contagem de boxes (para relatório)
    img_classes = {}
    img_boxes = {}
    kept_imgs = []
    missing_lbl = 0
    empty_lbl = 0

    for p in imgs:
        txt = label_path_for(p, root, labels_dir=args.labels_dir)
        cls_set, n_boxes = parse_classes_from_label(txt)
        if n_boxes == 0 and not Path(txt).exists():
            # sem label file
            if args.drop_missing_labels:
                missing_lbl += 1
                continue
        elif n_boxes == 0 and Path(txt).exists():
            empty_lbl += 1  # label vazio => background
        img_classes[p] = cls_set
        img_boxes[p] = n_boxes
        kept_imgs.append(p)

    imgs = kept_imgs
    if not imgs:
        print("[ERRO] Todas as imagens foram descartadas.")
        return

    total = len(imgs)
    n_val = max(1, int(total * args.val_ratio))

    # split
    if args.group_by_prefix > 0 and not args.stratify:
        # split por grupos para evitar near-duplicates
        groups = group_by_prefix(imgs, args.group_by_prefix)
        keys = list(groups.keys())
        random.shuffle(keys)
        val = []
        for k in keys:
            if len(val) >= n_val:
                break
            val.extend(groups[k])
        val = set(val[:n_val])
        train = [p for p in imgs if p not in val]
    elif args.stratify:
        # cobertura de classes no val, depois completa
        cand = imgs[:]
        random.shuffle(cand)
        val_list = greedy_cover_validation(cand, img_classes, n_val)
        val = set(val_list[:n_val])
        train = [p for p in imgs if p not in val]
    else:
        random.shuffle(imgs)
        val = set(imgs[:n_val])
        train = [p for p in imgs if p not in val]

    # relatório
    def split_stats(split_imgs):
        cls_hist = Counter()
        n_boxes = 0
        for p in split_imgs:
            cls_hist.update(img_classes.get(p, set()))
            n_boxes += img_boxes.get(p, 0)
        return cls_hist, n_boxes

    tr_hist, tr_boxes = split_stats(train)
    va_hist, va_boxes = split_stats(val)

    out_dir = Path(root, "splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list = sorted(train)
    val_list   = sorted(val)

    write_list(train_list, Path(out_dir, "train.txt"))
    write_list(val_list, Path(out_dir, "val.txt"))

    print(f"train: {len(train_list)}  val: {len(val_list)}  (seed={args.seed})")
    if missing_lbl or empty_lbl:
        print(f"backgrounds: sem .txt={missing_lbl}  txt vazio={empty_lbl} (use --drop-missing-labels para remover)")

    print_hist("train", tr_hist, len(train_list), tr_boxes)
    print_hist("val  ", va_hist, len(val_list), va_boxes)

    # YAML opcional (lista de arquivos absoluta → YOLO aceita)
    if args.yaml_out:
        names = None
        classes_yaml = Path(root, "classes.yaml")
        if classes_yaml.exists():
            try:
                names = yaml.safe_load(open(classes_yaml))["names"]  # type: ignore
            except Exception:
                pass
        data = {
            "names": names if names is not None else [],
            "train": str(Path(out_dir, "train.txt").resolve()),
            "val":   str(Path(out_dir, "val.txt").resolve()),
        }
        try:
            import yaml  # lazy
            with open(args.yaml_out, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            print("YAML salvo em:", args.yaml_out)
        except Exception as e:
            print("[WARN] Falha ao escrever YAML:", e)

if __name__ == "__main__":
    main()

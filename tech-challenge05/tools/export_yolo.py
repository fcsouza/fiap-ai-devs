import argparse, os, glob, random, yaml, pathlib

def list_images(root, exts=(".jpg", ".jpeg", ".png")):
    ims = []
    for e in exts:
        ims += glob.glob(os.path.join(root, f"*{e}"))
    return sorted(set(ims))

def ensure_splits(root, val_ratio=0.1, exts=(".jpg",".jpeg",".png")):
    """
    Gera ou reutiliza splits para um dataset no formato:
    root/
      images/            (ou images/train, images/val)
      labels/            (espelha as imagens)
      splits/train.txt, val.txt
    Retorna caminhos ABSOLUTOS para os txts.
    """
    root = os.path.abspath(root)
    images_root = os.path.join(root, "images")
    splits_dir  = os.path.join(root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_txt = os.path.join(splits_dir, "train.txt")
    val_txt   = os.path.join(splits_dir, "val.txt")

    # 1) Se já existem, só retorna
    if os.path.isfile(train_txt) and os.path.isfile(val_txt):
        return train_txt, val_txt

    # 2) Se existem subpastas train/val, use-as
    train_dir = os.path.join(images_root, "train")
    val_dir   = os.path.join(images_root, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train = list_images(train_dir, exts)
        val   = list_images(val_dir,   exts)
    else:
        # 3) Caso contrário, parta de images/ e faça split aleatório
        imgs = list_images(images_root, exts)
        if not imgs:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_root}")
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * float(val_ratio)))
        val = imgs[:n_val]
        train = imgs[n_val:]

    with open(train_txt, "w") as f: f.write("\n".join(train))
    with open(val_txt,   "w") as f: f.write("\n".join(val))
    return train_txt, val_txt

def main():
    ap = argparse.ArgumentParser(description="Gera data.yaml para YOLOv8, com 1 ou 2 datasets (sintético + real).")
    ap.add_argument("--syn_root", required=True, help="pasta do dataset sintético (ex.: synthetic_v2)")
    ap.add_argument("--real_root", default=None, help="(opcional) pasta do dataset real já em formato YOLO")
    ap.add_argument("--classes", required=True, help="classes.yaml CANÔNICO (do sintético)")
    ap.add_argument("--out_dir", required=True, help="pasta onde salvar data.yaml")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="split aleatório caso não exista train/val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Carrega classes oficiais (do sintético)
    with open(args.classes, "r") as f:
        names = yaml.safe_load(f)["names"]
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"Arquivo {args.classes} inválido (esperado chave 'names' com lista).")
    nc = len(names)

    # Garante/pega splits do sintético
    syn_train, syn_val = ensure_splits(args.syn_root, val_ratio=args.val_ratio)

    train_list = [syn_train]
    val_list   = [syn_val]

    # Se existir dataset real, também garante/pega splits e adiciona
    if args.real_root:
        real_train, real_val = ensure_splits(args.real_root, val_ratio=args.val_ratio)
        train_list.append(real_train)
        val_list.append(real_val)

    # Escreve data.yaml (sem 'path'; usa caminhos absolutos em listas)
    data_yaml = {
        "train": train_list,
        "val": val_list,
        "nc": nc,
        "names": names
    }
    out_yaml = os.path.join(args.out_dir, "data.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    # Também salva um classes.txt por conveniência
    with open(os.path.join(args.out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(names))

    print("✔ data.yaml salvo em:", out_yaml)
    print("  → train:", *train_list, sep="\n     ")
    print("  → val:  ", *val_list,   sep="\n     ")
    print("  nc:", nc)

if __name__ == "__main__":
    main()

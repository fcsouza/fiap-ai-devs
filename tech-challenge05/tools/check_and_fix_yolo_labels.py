import os, glob
from pathlib import Path

def clamp01(x): return max(0.0, min(1.0, x))

def load_classes(classes_path):
    with open(classes_path) as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def check_and_fix(label_path, n_classes):
    fixed = []
    bad = False
    with open(label_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            parts = ln.split()
            if len(parts) != 5:
                bad = True
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
            except Exception:
                bad = True
                continue
            if not (0 <= cid < n_classes):
                bad = True
                continue
            xc, yc, bw, bh = map(clamp01, (xc, yc, bw, bh))
            if bw <= 0 or bh <= 0:
                bad = True
                continue
            fixed.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(fixed))
    return bad

def main(base_path="synthetics/v2"):
    base = Path(base_path)
    classes_path = base / "classes.txt"
    labels_dir = base / "labels"
    classes = load_classes(classes_path)
    n_classes = len(classes)
    print(f"Classes carregadas: {n_classes}")
    # Busca recursiva por labels
    labs = glob.glob(str(labels_dir / "**/*.txt"), recursive=True)
    bads = 0
    for p in labs:
        if check_and_fix(p, n_classes):
            bads += 1
    print(f"[OK] Labels verificadas: {len(labs)}  corrigidas/filtradas: {bads}")

if __name__ == "__main__":
    main()
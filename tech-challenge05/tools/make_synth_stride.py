import argparse, os, random, math, json, io, re, glob
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from tqdm import tqdm
import yaml
import numpy as np

# ==========================
# Classes
# ==========================
DEFAULT_ICON_CLASSES = [
    line.strip() for line in """
    # ... (mantenha a lista completa de classes aqui)
    """.splitlines() if line.strip()
]

def load_classes(path=None) -> list[str]:
    if path:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo de classes não encontrado: {path}")
        with open(path, "r", encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
        if not classes:
            raise ValueError(f"Arquivo de classes está vazio: {path}")
        return classes
    return list(DEFAULT_ICON_CLASSES)

# ==========================
# Utils
# ==========================
def clean_name(name: str) -> str:
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r'\b(icon|icons|logo|logos|icon_azure|azure_icon|aws_icon|gcp_icon)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[_\-\s]+', '_', name).strip('_').lower()
    return name

def load_icons(icon_dir: str):
    icons = []
    for p in glob.glob(os.path.join(icon_dir, "**/*.png"), recursive=True):
        try:
            img = Image.open(p).convert("RGBA")
            raw = os.path.splitext(os.path.basename(p))[0]
            icons.append({ "path": p, "name": clean_name(raw), "display": raw.replace("_"," ").replace("-"," "), "img": img })
        except Exception:
            pass
    return icons

def rand_pos(w, h, iw, ih, margin=16):
    return (
        random.randint(margin, max(margin, w - iw - margin)),
        random.randint(margin, max(margin, h - ih - margin))
    )

def check_overlap(x, y, w, h, placed):
    for (px, py, pw, ph) in placed:
        if not (x + w < px or x > px + pw or y + h < py or y > py + ph):
            return True
    return False

def draw_dashed(draw, xy, width=2, fill=(0,0,0), dash=(8,8)):
    x1,y1,x2,y2 = xy
    total = math.hypot(x2-x1, y2-y1)
    if total <= 0: return
    dx,dy = (x2-x1)/total, (y2-y1)/total
    d_on, d_off = dash
    dist = 0.0
    while dist < total:
        s, e = dist, min(dist + d_on, total)
        sx,sy = x1 + dx*s, y1 + dy*s
        ex,ey = x1 + dx*e, y1 + dy*e
        draw.line((sx,sy,ex,ey), fill=fill, width=width)
        dist += d_on + d_off

def get_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Helvetica.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    random.shuffle(font_paths)
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def random_label_style(label):
    style = random.choice(['upper','lower','capitalize','title','break','plain'])
    if style == 'upper': return label.upper()
    if style == 'lower': return label.lower()
    if style == 'capitalize': return label.capitalize()
    if style == 'title': return label.title()
    if style == 'break':
        parts = label.split()
        if len(parts)>1:
            i = random.randint(1, len(parts)-1)
            return ' '.join(parts[:i])+'\n'+' '.join(parts[i:])
    return label

def paste_with_shadow(canvas: Image.Image, icon: Image.Image, xy, alpha=120):
    x,y = xy
    shadow = Image.new("RGBA", icon.size, (0,0,0,0))
    sh = Image.new("RGBA", icon.size, (0,0,0,alpha))
    shadow.paste(sh, (2,2))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
    canvas.alpha_composite(shadow, (x,y))
    canvas.alpha_composite(icon, (x,y))

def jitter_icon(pil_rgba: Image.Image):
    arr = np.array(pil_rgba).astype(np.float32)
    mul = np.clip(np.random.normal(1.0, 0.05, size=(1,1,4)), 0.85, 1.15)
    mul[0,0,3] = 1.0
    arr = np.clip(arr * mul, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    if random.random() < 0.5:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img

def draw_arrow(draw: ImageDraw.ImageDraw, p1, p2, width=3, fill=(0,0,0)):
    draw.line((*p1,*p2), width=width, fill=fill)
    ang = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
    L = 10 + random.randint(0,6)
    a1 = (p2[0]-L*math.cos(ang-0.5), p2[1]-L*math.sin(ang-0.5))
    a2 = (p2[0]-L*math.cos(ang+0.5), p2[1]-L*math.sin(ang+0.5))
    draw.polygon([p2, a1, a2], fill=fill)

def place_text(draw, text, box_xywh, align='bottom'):
    x,y,w,h = box_xywh
    lines = text.split('\n')
    font = get_font(random.randint(18, 34))
    sizes = [draw.textbbox((0,0), ln, font=font) for ln in lines]
    widths = [s[2]-s[0] for s in sizes]
    heights = [s[3]-s[1] for s in sizes]
    tw, th = max(widths), sum(heights)+2*(len(lines)-1)
    if align == 'top':
        tx = x + (w - tw)//2; ty = y - th - 4
    elif align == 'left':
        tx = x - tw - 6; ty = y + (h - th)//2
    elif align == 'right':
        tx = x + w + 6; ty = y + (h - th)//2
    else:
        tx = x + (w - tw)//2; ty = y + h + 2
    cy = ty
    for ln in lines:
        draw.text((tx, cy), ln, fill=(0,0,0), font=font)
        bb = draw.textbbox((tx, cy), ln, font=font)
        cy += (bb[3]-bb[1]) + 2

def jpeg_compress(pil_img: Image.Image, q_min=40, q_max=95):
    q = random.randint(q_min, q_max)
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    out = Image.open(buf).convert("RGBA")
    return out

def add_noise(pil_img: Image.Image, std=6.0):
    arr = np.array(pil_img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")

def motion_blur(pil_img: Image.Image, k=5):
    if k<=1: return pil_img
    kernel = np.zeros((k,k))
    kernel[int((k-1)/2), :] = 1.0
    kernel /= k
    arr = np.array(pil_img.convert("RGB"))
    arr = cv2_filter(arr, kernel)
    out = Image.fromarray(arr).convert("RGBA")
    return out

def cv2_filter(arr, kernel):
    import cv2
    return cv2.filter2D(arr, -1, kernel)

def perspective_warp(pil_img: Image.Image, max_warp=0.06):
    w,h = pil_img.size
    dx = int(w*max_warp*random.uniform(-1,1))
    dy = int(h*max_warp*random.uniform(-1,1))
    src = [(0,0), (w,0), (w,h), (0,h)]
    dst = [(0+dx,0), (w+dx,0+dy), (w, h), (0, h+dy)]
    return pil_img.transform((w,h), Image.QUAD, sum(dst,()), Image.BICUBIC)

def occlude_rect(draw, box, color=(255,255,255,200)):
    x1,y1,x2,y2 = box
    w = x2-x1; h = y2-y1
    ow = int(w * random.uniform(0.2, 0.5))
    oh = int(h * random.uniform(0.2, 0.5))
    ox = random.randint(x1, x2 - ow)
    oy = random.randint(y1, y2 - oh)
    draw.rectangle([ox,oy,ox+ow,oy+oh], fill=color)

# ==========================
# LLM (opcional) + class map
# ==========================
def canonicalize_class_name(value, classes_set, normalized_map):
    if not value: return None
    candidate = value.strip()
    if candidate in classes_set: return candidate
    return normalized_map.get(clean_name(candidate))

def classify_ollama(icon_name, classes):
    import ollama
    classes_prompt = "\n".join(classes)
    prompt = (
        "Classifique o ícone de serviço de nuvem informado na lista de classes fornecida.\n"
        "Responda usando exatamente um dos nomes abaixo (sensível a sublinhados).\n"
        "Classes disponíveis:\n"
        f"{classes_prompt}\n\n"
        f"Ícone: '{icon_name}'.\n"
        "Forneça somente o nome da classe escolhida."
    )
    resp = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    return resp['message']['content'].strip()

def load_class_map(class_map_json, use_llm, icons, classes, default_class):
    classes_set = set(classes)
    normalized_map = {clean_name(cls): cls for cls in classes}
    icon_to_class = {}
    if class_map_json and os.path.exists(class_map_json):
        with open(class_map_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            canonical = canonicalize_class_name(v, classes_set, normalized_map)
            if not canonical: continue
            icon_to_class[clean_name(k)] = canonical
    for ic in icons:
        key = ic["name"]
        if key in icon_to_class: continue
        canonical = canonicalize_class_name(ic["name"], classes_set, normalized_map)
        if not canonical:
            canonical = canonicalize_class_name(ic.get("display", ""), classes_set, normalized_map)
        if canonical:
            icon_to_class[key] = canonical
    if use_llm:
        print(f"[INFO] Classificando {len(icons)} ícones com Llama3 (Ollama)...")
        for ic in tqdm(icons, desc="Classificando ícones com Llama3"):
            key = ic["name"]
            if key in icon_to_class: continue
            try:
                predicted = classify_ollama(ic["display"], classes)
            except Exception:
                predicted = ""
            canonical = canonicalize_class_name(predicted, classes_set, normalized_map)
            if not canonical:
                canonical = default_class
            icon_to_class[key] = canonical
    for ic in icons:
        icon_to_class.setdefault(ic["name"], default_class)
    return icon_to_class

# ==========================
# Balanceamento dos ícones
# ==========================
def build_icon_usage_list(icons, icon_to_class, min_uses=10):
    icon_list = []
    for ic in icons:
        key = ic["name"]
        if key in icon_to_class:
            icon_list.extend([ic] * min_uses)
    random.shuffle(icon_list)
    return icon_list

# ==========================
# Main
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--out", type=str, default="synthetic")
    ap.add_argument("--icons", type=str, default="icons")
    ap.add_argument("--canvas", type=str, default="1024x768")
    ap.add_argument("--seed", type=int, default=42)
    # domínio real (aumentações)
    ap.add_argument("--skew_max_deg", type=float, default=7.0)
    ap.add_argument("--persp_max", type=float, default=0.06)
    ap.add_argument("--jpeg_min", type=int, default=40)
    ap.add_argument("--jpeg_max", type=int, default=95)
    ap.add_argument("--noise_std", type=float, default=6.0)
    ap.add_argument("--blur_sigma", type=float, default=0.8)
    ap.add_argument("--p_motion_blur", type=float, default=0.35)
    ap.add_argument("--p_perspective", type=float, default=0.6)
    ap.add_argument("--p_jpeg", type=float, default=0.9)
    ap.add_argument("--p_noise", type=float, default=0.7)
    ap.add_argument("--p_downup", type=float, default=0.6)
    ap.add_argument("--p_occlude", type=float, default=0.25)
    # cenas
    ap.add_argument("--min_nodes", type=int, default=6)
    ap.add_argument("--max_nodes", type=int, default=16)
    ap.add_argument("--p_negative", type=float, default=0.12)
    # rotulagem
    ap.add_argument("--classes_file", type=str, default="", help="Arquivo com classes (uma por linha).")
    ap.add_argument("--class_map_json", type=str, default="", help="JSON que mapeia nomes de ícones para classes.")
    ap.add_argument("--stride_map_json", type=str, default="", help=argparse.SUPPRESS)
    ap.add_argument("--default_class", type=str, default="missing", help="Classe padrão caso o ícone não seja mapeado.")
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--min_uses", type=int, default=10, help="Mínimo de vezes que cada ícone será usado.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    W, H = map(int, args.canvas.lower().split("x"))
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    # 0) Classes
    classes = load_classes(args.classes_file or None)
    if not classes:
        print("Nenhuma classe configurada.")
        return
    name_to_id = {n: i for i, n in enumerate(classes)}
    default_class = args.default_class if args.default_class in name_to_id else classes[0]
    if args.default_class and args.default_class not in name_to_id:
        print(f"[AVISO] Classe padrão '{args.default_class}' não encontrada. Usando '{default_class}'.")

    # 1) Ícones + mapa de classes
    icons = load_icons(args.icons)
    if not icons:
        print("Nenhum ícone PNG encontrado em", args.icons)
        return
    map_json = args.class_map_json or args.stride_map_json
    icon_to_class = load_class_map(map_json, args.use_llm, icons, classes, default_class)

    # bucket por classe para balancear
    class_to_icons = {c: [] for c in classes}
    for ic in icons:
        mapped = icon_to_class.get(ic["name"])
        if mapped in class_to_icons:
            class_to_icons[mapped].append(ic)
    available_classes = [c for c, items in class_to_icons.items() if items]
    if not available_classes:
        print("Nenhum ícone foi associado às classes configuradas.")
        return

    # Garante que todos os ícones sejam usados várias vezes
    icon_usage_list = build_icon_usage_list(icons, icon_to_class, min_uses=args.min_uses)
    total_imgs = max(args.num, len(icon_usage_list))

    for i in tqdm(range(total_imgs), desc="Gerando diagramas sintéticos"):
        img = Image.new("RGBA", (W,H), (255,255,255,255))
        draw = ImageDraw.Draw(img)
        # “painéis”/cards tipo Azure/AWS
        if random.random() < 0.9:
            for _ in range(random.randint(1,3)):
                px = random.randint(40, W-400)
                py = random.randint(40, H-300)
                pw = random.randint(300, 700)
                ph = random.randint(220, 480)
                color = (0,0,0,60) if random.random()<0.5 else (0,0,0,0)
                if random.random() < 0.5:
                    draw.rectangle([px,py,px+pw,py+ph], outline=(140,140,140,255), width=2)
                else:
                    draw_dashed(draw, (px,py,px+pw,py), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px,py+ph,px+pw,py+ph), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px,py,px,py+ph), width=2, fill=(160,160,160))
                    draw_dashed(draw, (px+pw,py,px+pw,py+ph), width=2, fill=(160,160,160))
                if color[3]>0:
                    ImageDraw.Draw(img).rectangle([px+1,py+1,px+pw-1,py+ph-1], fill=color)
        is_negative = random.random() < args.p_negative
        placed = []
        boxes = []
        n_nodes = 0 if is_negative else random.randint(args.min_nodes, args.max_nodes)
        centers = []
        for _ in range(n_nodes):
            if icon_usage_list:
                ii = icon_usage_list.pop()
                cls_name = icon_to_class[ii["name"]]
            else:
                cls_name = random.choice(available_classes)
                icon_list = class_to_icons[cls_name]
                ii = random.choice(icon_list)
            icon = jitter_icon(ii["img"])
            scale = random.uniform(0.65, 1.55)
            iw, ih = icon.size
            icon_r = icon.resize((int(iw*scale), int(ih*scale)), resample=Image.BICUBIC)
            iw2, ih2 = icon_r.size
            placed_ok = False
            for _try in range(80):
                x, y = rand_pos(W, H, iw2, ih2 + 40)
                if not check_overlap(x, y, iw2, ih2, placed):
                    placed_ok = True; break
            if not placed_ok: continue
            placed.append((x,y,iw2,ih2))
            paste_with_shadow(img, icon_r, (x,y))
            if random.random() < args.p_occlude:
                occlude_rect(draw, (x,y,x+iw2,y+ih2), color=(255,255,255, random.randint(120,200)))
            label = random_label_style(ii["display"])
            pos = random.choice(["top","bottom","left","right"])
            if pos=="top":
                place_text(draw, label, (x,y,iw2,ih2), align='top')
            elif pos=="left":
                place_text(draw, label, (x,y,iw2,ih2), align='left')
            elif pos=="right":
                place_text(draw, label, (x,y,iw2,ih2), align='right')
            else:
                place_text(draw, label, (x,y,iw2,ih2), align='bottom')
            boxes.append((x, y, x+iw2, y+ih2, cls_name))
            centers.append((x+iw2//2, y+ih2//2))
        n_conn = random.randint(max(1, len(centers)//2), len(centers)+2)
        for _ in range(n_conn):
            if len(centers) < 2: break
            a, b = random.sample(centers, 2)
            width = random.randint(2,5)
            if random.random() < 0.5:
                draw.line((*a,*b), width=width, fill=(0,0,0))
            else:
                draw_dashed(draw, (*a,*b), width=width, fill=(0,0,0), dash=(10,8))
            if random.random() < 0.6:
                draw_arrow(draw, a, b, width=max(2,width-1), fill=(0,0,0))
        # Aumentações
        if random.random() < 0.7:
            img = img.rotate(random.uniform(-args.skew_max_deg, args.skew_max_deg), resample=Image.BICUBIC, expand=False, fillcolor=(255,255,255,0))
        if random.random() < args.p_perspective:
            img = perspective_warp(img, max_warp=args.persp_max)
        if random.random() < args.p_downup:
            s = random.uniform(0.6, 0.9)
            w2,h2 = int(W*s), int(H*s)
            img = img.resize((w2,h2), Image.BICUBIC).resize((W,H), Image.BICUBIC)
        if random.random() < 0.7:
            if random.random() < args.p_motion_blur:
                k = random.choice([3,5,7])
                img = motion_blur(img, k=k)
            else:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, args.blur_sigma)))
        if random.random() < args.p_noise:
            img = add_noise(img, std=args.noise_std)
        if random.random() < args.p_jpeg:
            img = jpeg_compress(img, q_min=args.jpeg_min, q_max=args.jpeg_max)
        img_path = os.path.join(args.out, "images", f"syn_{i:06d}.jpg")
        img.convert("RGB").save(img_path, quality=95)
        yolo_lines = []
        for (x1,y1,x2,y2,cls) in boxes:
            cid = name_to_id[cls]
            xc = ((x1+x2)/2.0)/W
            yc = ((y1+y2)/2.0)/H
            bw = (x2-x1)/W
            bh = (y2-y1)/H
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(args.out, "labels", f"syn_{i:06d}.txt"), "w") as f:
            f.write("\n".join(yolo_lines))

    with open(os.path.join(args.out, "classes.yaml"), "w") as f:
        yaml.safe_dump({"names": classes}, f, sort_keys=False)

    print("[OK] Dataset sintético gerado em:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
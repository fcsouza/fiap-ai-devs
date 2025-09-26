# tools/shrink_images.py
import argparse, os, io, sys, math, pathlib, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, PngImagePlugin
from tqdm import tqdm

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root):
    root = pathlib.Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS]

def ensure_dir(p):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

def has_meaningful_alpha(img):
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
        # amostrar para acelerar
        w, h = img.size
        step = max(1, min(w, h) // 800)  # amostra ~800 px no menor lado
        data = alpha.crop((0,0,w,h)).resize((w//step, h//step))
        extrema = data.getextrema()  # (min, max)
        # se 0<alpha<255 existir, há semitransparência relevante
        return extrema[0] < 255 and extrema[1] > 0
    return False

def drop_alpha_if_empty(img, bg=(255,255,255)):
    if img.mode in ("RGBA","LA"):
        alpha = img.getchannel("A")
        mn,mx = alpha.getextrema()
        if mx == 255:
            # totalmente opaco → remove alpha
            return img.convert("RGB")
        if mn == 255:
            return img.convert("RGB")
        if mn == 0 and mx == 0:
            # totalmente transparente (raro) → preenche fundo branco e remove alpha
            bg_img = Image.new("RGB", img.size, bg)
            bg_img.paste(img, mask=alpha)
            return bg_img
    return img

def downscale_if_needed(img, max_side):
    if max_side <= 0:
        return img
    w,h = img.size
    m = max(w,h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new = (max(1,int(w*scale)), max(1,int(h*scale)))
    return img.resize(new, Image.LANCZOS)

def save_png_quantized(img, out_path, colors=256, compress_level=9, optimize=True):
    # Converte para paleta (ótimo para diagramas/ícones) e salva com compressão máxima
    work = img
    if img.mode not in ("RGB","L","P"):
        work = img.convert("RGBA")
    try:
        q = work.quantize(colors=colors, method=2)  # 2=FASTOCTREE
    except Exception:
        q = work.convert("P", palette=Image.ADAPTIVE, colors=colors)
    ensure_dir(out_path)
    q.save(out_path, format="PNG", optimize=optimize, compress_level=compress_level)
    return os.path.getsize(out_path)

def save_jpg(img, out_path, quality=90, subsampling="4:4:4"):
    # Só se não tiver alpha relevante
    ensure_dir(out_path)
    rgb = img.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=quality, optimize=True, subsampling=subsampling)
    return os.path.getsize(out_path)

def pick_smaller(candidates, fallback_path):
    # candidates: list[(path,size)]
    # mantém o menor, remove os demais; se algo falhar, mantém fallback
    cand = [(p,s) for p,s in candidates if p and os.path.exists(p)]
    if not cand:
        return fallback_path
    best = min(cand, key=lambda x: x[1])[0]
    for p,_ in cand:
        if p != best:
            try: os.remove(p)
            except: pass
    return best

def save_optimized(img, out_path, ext):
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return save_jpg(img, out_path, quality=85, subsampling="4:4:4")  # ajuste qualidade se quiser
    elif ext == ".png":
        return save_png_quantized(img, out_path, colors=256, compress_level=9, optimize=True)
    else:
        img.save(out_path)
        return os.path.getsize(out_path)

def cleanup_tempfiles(cands):
    for p, _ in cands:
        if os.path.exists(p):
            os.remove(p)

def process_one(src_path, out_root, max_side, png_colors, try_jpeg_for_opaque, keep_ext=False):
    src_path = pathlib.Path(src_path)
    rel = src_path.relative_to(src_path.anchor) if out_root == "__INPLACE__" else src_path.relative_to(src_path.parents[0])
    # melhor: replicar árvore relativa a uma base comum
    if out_root == "__INPLACE__":
        out_dir = src_path.parent
    else:
        # tenta preservar estrutura a partir da pasta raiz que o usuário passou
        out_dir = pathlib.Path(out_root) / src_path.parent.name
    dst_stem = out_dir / src_path.stem

    try:
        with Image.open(src_path) as im0:
            im = im0.convert("RGBA") if im0.mode in ("P","LA") else im0.copy()
    except Exception as e:
        return (str(src_path), 0, 0, f"open_error:{e}")

    orig_size = os.path.getsize(src_path)
    img = downscale_if_needed(im, max_side)
    img = drop_alpha_if_empty(img)

    # Sempre gera uma versão PNG quantizada (boa p/ diagramas)
    tmp_png = str(dst_stem) + ".__tmp.png"
    try:
        png_size = save_png_quantized(img, tmp_png, colors=png_colors)
    except Exception as e:
        return (str(src_path), orig_size, 0, f"png_error:{e}")
    cands = [(tmp_png, png_size)]

    # JPEG (somente se imagem for opaca e potencialmente fotográfica)
    if try_jpeg_for_opaque and img.mode != "RGBA":
        tmp_jpg = str(dst_stem) + ".__tmp.jpg"
        try:
            jpg_size = save_jpg(img, tmp_jpg, quality=90)
            cands.append((tmp_jpg, jpg_size))
        except Exception:
            pass

    # Mantém o menor e renomeia para extensão final coerente
    if keep_ext:
        final_path = str(dst_stem) + src_path.suffix.lower()
        try:
            size_opt = save_optimized(img, final_path, src_path.suffix)
        except Exception as e:
            return (str(src_path), orig_size, 0, f"opt_error:{e}")
        # Se não ficou menor, mantém o original
        if size_opt >= orig_size:
            os.remove(final_path)
            if out_root == "__INPLACE__":
                return (str(src_path), orig_size, orig_size, "kept_original")
            else:
                final_path = str(pathlib.Path(out_dir) / src_path.name)
                ensure_dir(final_path)
                shutil.copy2(src_path, final_path)
                return (str(src_path), orig_size, orig_size, "copied_original")
        cleanup_tempfiles(cands)
        new_size = size_opt
        return (str(src_path), orig_size, new_size, "ok")
    else:
        best_tmp = pick_smaller(cands, tmp_png)
        if best_tmp.endswith(".png"):
            final_path = str(dst_stem) + ".png"
        else:
            final_path = str(dst_stem) + os.path.splitext(best_tmp)[1]

    ensure_dir(final_path)
    try:
        os.replace(best_tmp, final_path)
    except Exception:
        # fallback: copy
        shutil.copy2(best_tmp, final_path)
        os.remove(best_tmp)

    new_size = os.path.getsize(final_path)
    # Se piorou, mantém original (copia)
    if new_size >= orig_size:
        try:
            os.remove(final_path)
        except: pass
        if out_root == "__INPLACE__":
            # já está no mesmo lugar, não faz nada
            return (str(src_path), orig_size, orig_size, "kept_original")
        else:
            final_path = str(pathlib.Path(out_dir) / src_path.name)
            ensure_dir(final_path)
            shutil.copy2(src_path, final_path)
            return (str(src_path), orig_size, orig_size, "copied_original")

    return (str(src_path), orig_size, new_size, "ok")

def main():
    ap = argparse.ArgumentParser(description="Otimizador de imagens grandes (PNG/diagramas).")
    ap.add_argument("--inp", required=True, help="Pasta de entrada (varre recursivo).")
    ap.add_argument("--out", default="__INPLACE__", help="Pasta de saída. Use '__INPLACE__' para sobrescrever.")
    ap.add_argument("--max-side", type=int, default=2600, help="Maior lado permitido (0 = não redimensiona).")
    ap.add_argument("--png-colors", type=int, default=256, help="Cores da paleta PNG (128~256 recomendável).")
    ap.add_argument("--jpeg-if-opaque", action="store_true", help="Considerar JPEG (se imagem opaca).")
    ap.add_argument("--workers", type=int, default=4, help="Processos em paralelo.")
    ap.add_argument("--keep-ext", action="store_true", help="Mantém a extensão original do arquivo.")

    args = ap.parse_args()

    imgs = list_images(args.inp)
    if not imgs:
        print("[ERRO] Nenhuma imagem encontrada.")
        sys.exit(1)

    total_orig = 0
    total_new  = 0
    ok, kept, errors = 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, str(p), args.out, args.max_side, args.png_colors,
                          args.jpeg_if_opaque, args.keep_ext) for p in imgs]
        for f in tqdm(as_completed(futs), total=len(futs), desc="Otimizando"):
            try:
                src, so, sn, status = f.result()
                total_orig += so
                total_new  += (sn if sn else so)
                if status in ("ok", "copied_original", "kept_original"):
                    ok += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1

    saved_mb = (total_orig - total_new) / (1024*1024.0)
    orig_mb  = total_orig / (1024*1024.0)
    new_mb   = total_new  / (1024*1024.0)
    ratio = (1.0 - (total_new / total_orig)) * 100 if total_orig > 0 else 0.0
    print("\n[RESUMO]")
    print(f"  arquivos processados : {len(imgs)}  ok={ok}  erros={errors}")
    print(f"  tamanho original     : {orig_mb:.2f} MB")
    print(f"  tamanho final        : {new_mb:.2f} MB")
    print(f"  economia             : {saved_mb:.2f} MB ({ratio:.1f}%)")

if __name__ == "__main__":
    main()

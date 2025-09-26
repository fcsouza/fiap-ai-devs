import argparse
import json
import os
import glob
import sys
from pathlib import Path
from tqdm import tqdm

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import make_synth_stride as synth
import ollama
import Levenshtein

def normalize_icon_name(name):
    return name.lower().replace("-", "_").replace(" ", "_")

def classify_ollama_batch(icon_names, classes):
    norm_classes = [normalize_icon_name(c) for c in classes]
    prompt = (
        "Classifique cada um dos seguintes ícones em uma das categorias: "
        f"{', '.join(norm_classes)}.\n"
        "Responda no formato: <nome_do_ícone>: <categoria>\n"
        "Use apenas uma das categorias fornecidas, sem criar novas.\n"
        "Nunca use 'unknown', 'none', ou qualquer valor fora das categorias listadas.\n"
        "Se não tiver certeza, escolha a categoria mais próxima ou mais provável.\n"
        "Ícones para classificar:\n"
    )
    prompt += "\n".join([f"{name}:" for name in icon_names])
    response = ollama.chat(model='llama3', messages=[{"role": "user", "content": prompt}])
    result = {}
    classes_set = set(norm_classes)
    for line in response['message']['content'].splitlines():
        line = line.strip().lstrip("*").strip()
        if ":" in line:
            name, cat = line.split(":", 1)
            name = name.strip()
            cat = cat.strip().lower().replace(" ", "_")
            if cat not in classes_set:
                cat = min(classes_set, key=lambda c: Levenshtein.distance(cat, c))
            result[name] = cat
    return result

def build_mapping(icon_dir, classes, use_llm=False, batch_size=30, max_retries=3):
    classes_set = set(classes)
    normalized_map = {synth.clean_name(cls): cls for cls in classes}
    icon_to_class = {}
    unmatched = []
    icon_paths = sorted(glob.glob(os.path.join(icon_dir, "**/*.png"), recursive=True))
    icon_names = []
    for icon_path in icon_paths:
        raw_name = os.path.splitext(os.path.basename(icon_path))[0]
        icon_names.append(normalize_icon_name(raw_name))
        key = synth.clean_name(raw_name)
        canonical = synth.canonicalize_class_name(raw_name, classes_set, normalized_map)
        if not canonical:
            canonical = synth.canonicalize_class_name(key, classes_set, normalized_map)
        if not canonical:
            canonical = synth.canonicalize_class_name(raw_name.replace("_", " "), classes_set, normalized_map)
        if not canonical:
            unmatched.append(raw_name)
        else:
            icon_to_class[key] = canonical

    if use_llm and unmatched:
        norm_classes = [normalize_icon_name(c) for c in classes]
        retries = 0
        still_unmatched = unmatched
        while still_unmatched and retries < max_retries:
            print(f"[INFO] Tentativa {retries+1} de classificação LLM para {len(still_unmatched)} ícones...")
            for i in tqdm(range(0, len(still_unmatched), batch_size), desc=f"Classificando com LLM (tentativa {retries+1})"):
                batch = still_unmatched[i:i+batch_size]
                batch_result = classify_ollama_batch(batch, classes)
                for orig_name in batch:
                    predicted = batch_result.get(orig_name, "")
                    if predicted in norm_classes:
                        idx = norm_classes.index(predicted)
                        canonical = classes[idx]
                    else:
                        canonical = None
                    if canonical:
                        icon_to_class[synth.clean_name(orig_name)] = canonical
            still_unmatched = [name for name in still_unmatched if synth.clean_name(name) not in icon_to_class]
            retries += 1
        unmatched = still_unmatched
    else:
        unmatched = [name for name in unmatched if synth.clean_name(name) not in icon_to_class]

    return icon_to_class, unmatched

def reclassify_unmatched(json_path, icon_dir, classes, use_llm=True, batch_size=30, max_retries=2):
    # Carrega o mapeamento anterior
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Lista todos os ícones
    icon_paths = sorted(glob.glob(os.path.join(icon_dir, "**/*.png"), recursive=True))
    all_icon_names = [os.path.splitext(os.path.basename(p))[0] for p in icon_paths]
    # Descobre os que não estão mapeados
    mapped_keys = set(mapping.keys())
    unmatched = [name for name in all_icon_names if name not in mapped_keys or not mapping[name]]
    print(f"[INFO] {len(unmatched)} ícones sem correspondência para reclassificar.")
    # Reclassifica só os não mapeados
    if use_llm and unmatched:
        norm_classes = [normalize_icon_name(c) for c in classes]
        retries = 0
        still_unmatched = unmatched
        while still_unmatched and retries < max_retries:
            print(f"[INFO] Tentativa {retries+1} de classificação LLM para {len(still_unmatched)} ícones...")
            for i in tqdm(range(0, len(still_unmatched), batch_size), desc=f"Classificando com LLM (tentativa {retries+1})"):
                batch = still_unmatched[i:i+batch_size]
                batch_result = classify_ollama_batch(batch, classes)
                for orig_name in batch:
                    predicted = batch_result.get(orig_name, "")
                    if predicted in norm_classes:
                        idx = norm_classes.index(predicted)
                        canonical = classes[idx]
                    else:
                        canonical = None
                    if canonical:
                        mapping[orig_name] = canonical
            still_unmatched = [name for name in still_unmatched if not mapping.get(name)]
            retries += 1
    # Salva o novo JSON atualizado
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] Atualizado {json_path} com novas classificações.")

def main():
    parser = argparse.ArgumentParser(description="Gera um class_map_json a partir das classes disponíveis e dos ícones encontrados.")
    parser.add_argument("--icons", default="icons", help="Diretório com os ícones em PNG.")
    parser.add_argument("--classes_file", required=False, default="", help="Arquivo com classes (uma por linha).")
    parser.add_argument("--out", required=True, help="Caminho para salvar o JSON gerado.")
    parser.add_argument("--default_class", default="", help="Classe para ícones sem correspondência quando LLM desativado.")
    parser.add_argument("--use_llm", action="store_true", help="Tentar classificar ícones sem correspondência via Llama3/ollama.")
    parser.add_argument("--reclassify", action="store_true", help="Reclassifica apenas os ícones não mapeados no JSON de saída.")
    args = parser.parse_args()

    classes = synth.load_classes(args.classes_file or None)
    if not classes:
        raise ValueError("Nenhuma classe encontrada.")

    if args.reclassify:
        reclassify_unmatched(args.out, args.icons, classes, use_llm=args.use_llm)
        return

    mapping, unmatched = build_mapping(args.icons, classes, use_llm=args.use_llm)

    default_class = args.default_class
    if default_class and default_class not in classes:
        raise ValueError(f"Classe padrão '{default_class}' não está na lista de classes.")
    if default_class:
        for icon_name in unmatched:
            mapping.setdefault(icon_name, default_class)
    elif unmatched:
        print(f"[AVISO] {len(unmatched)} ícones sem correspondência e sem classe padrão definida: {unmatched[:10]}...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    total = len(glob.glob(os.path.join(args.icons, "**/*.png"), recursive=True))
    print(f"[OK] Gerado {len(mapping)} mapeamentos (de {total} ícones).")
    if unmatched:
        print(f"[{len(unmatched)} sem correspondência]")

if __name__ == "__main__":
    main()
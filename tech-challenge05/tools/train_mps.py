# train_mps.py
import os, time, yaml
from ultralytics import YOLO

# aponte para o seu data.yaml combinado local
DATA_YAML = '../optmized/combined/data.yaml'   # <-- ajuste

# pasta dos runs e nome do experimento
RUNS_DIR = 'runs'                  # <-- ajuste
RUN_NAME = 'yolo_syn_real_mps_s'

# modelo base (pode usar 'yolov8m.pt'; se instável, troque p/ 'yolov8s.pt')
BASE_MODEL = 'yolov8s.pt'
LAST_PT = f'{RUNS_DIR}/detect/{RUN_NAME}/weights/last.pt'

# configs robustas p/ MPS (sem AMP, poucos workers)
train_cfg = dict(
    data=DATA_YAML,
    imgsz=640,       # 896 tende a aquecer/instabilizar no MPS; suba se estiver estável
    epochs=60,
    batch=12,         # aumente se ficar estável (12/16)
    device='mps',
    workers=0,       # 0-2 no MPS para evitar travas
    amp=False,       # MPS não tem AMP estável
    mosaic=0, mixup=0, copy_paste=0, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=3, translate=0.06, scale=0.9, perspective=0.0,
    cache='disk',     # evite 'disk' se o SO estiver matando por IO
    val=False,
    project=RUNS_DIR,
    name=RUN_NAME,
    exist_ok=True,
    resume=os.path.exists(LAST_PT),
    save_period=1,
    plots=False
)

def main():
    retries = 5
    for attempt in range(retries):
        try:
            model = YOLO(LAST_PT) if os.path.exists(LAST_PT) else YOLO(BASE_MODEL)
            model.train(**train_cfg)
            print("[OK] treino finalizado.")
            return
        except KeyboardInterrupt:
            print("[INFO] Interrompido. Você pode rodar de novo para retomar do last.pt.")
            return
        except Exception as e:
            print(f"[MPS][tentativa {attempt+1}] crash: {e}")
            time.sleep(10)  # espera e tenta retomar
    print("[ERRO] excedeu tentativas. Verifique logs do último run em:", f"{RUNS_DIR}/detect/{RUN_NAME}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import importlib
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    print("Testando importações dos módulos...")

    modules_to_test = [
        "cv2",
        "numpy",
        "mediapipe",
        "tensorflow",
        "keras",
        "sklearn",
        "matplotlib",
        "seaborn",
        "pandas",
        "PIL",
        "deepface",
        "ultralytics",
        "torch",
        "torchvision",
    ]

    failed_imports = []

    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)

    if failed_imports:
        print(f"\n⚠️  {len(failed_imports)} módulos não puderam ser importados:")
        for module in failed_imports:
            print(f"   - {module}")
        return False
    else:
        print("\n✅ Todas as importações foram bem-sucedidas!")
        return True


def test_src_modules():
    print("\nTestando módulos do projeto...")

    src_modules = [
        "src.utils",
        "src.video_processor",
        "src.face_detector",
        "src.emotion_analyzer",
        "src.activity_detector",
        "src.anomaly_detector",
        "src.report_generator",
    ]

    failed_modules = []

    for module_name in src_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_modules.append(module_name)

    if failed_modules:
        print(
            f"\n⚠️  {len(failed_modules)} módulos do projeto não puderam ser importados:"
        )
        for module in failed_modules:
            print(f"   - {module}")
        return False
    else:
        print("\n✅ Todos os módulos do projeto foram importados com sucesso!")
        return True


def test_video_file():
    print("\nTestando arquivo de vídeo...")

    video_path = "data/video.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Arquivo de vídeo não encontrado: {video_path}")
        print("   Certifique-se de que o arquivo video.mp4 está no diretório data/")
        return False

    try:
        from src.utils import get_video_info

        video_info = get_video_info(video_path)

        print(f"✅ Arquivo de vídeo encontrado: {video_path}")
        print(f"   - Duração: {video_info['duration']:.2f} segundos")
        print(f"   - FPS: {video_info['fps']:.2f}")
        print(f"   - Frames: {video_info['frame_count']:,}")
        print(f"   - Resolução: {video_info['width']}x{video_info['height']}")

        return True

    except Exception as e:
        print(f"❌ Erro ao ler arquivo de vídeo: {e}")
        return False


def test_directories():
    print("\nTestando criação de diretórios...")

    directories = ["output", "temp"]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Diretório {directory}/ criado/verificado")
        except Exception as e:
            print(f"❌ Erro ao criar diretório {directory}/: {e}")
            return False

    return True


def test_models():
    print("\nTestando modelos de IA...")

    try:
        from ultralytics import YOLO

        print("Baixando modelo YOLO...")
        model = YOLO("yolov8n.pt")
        print("✅ Modelo YOLO carregado com sucesso")

    except Exception as e:
        print(f"❌ Erro ao carregar modelo YOLO: {e}")
        return False

    try:
        from deepface import DeepFace

        print("Verificando modelos DeepFace...")
        models = DeepFace.build_model("Emotion")
        print("✅ Modelos DeepFace disponíveis")

    except Exception as e:
        print(f"❌ Erro ao carregar modelos DeepFace: {e}")
        return False

    return True


def test_basic_functionality():
    print("\nTestando funcionalidade básica...")

    try:
        from src.activity_detector import ActivityDetector
        from src.anomaly_detector import AnomalyDetector
        from src.emotion_analyzer import EmotionAnalyzer
        from src.face_detector import FaceDetector

        print("Inicializando detectores...")

        face_detector = FaceDetector()
        print("✅ FaceDetector inicializado")

        emotion_analyzer = EmotionAnalyzer()
        print("✅ EmotionAnalyzer inicializado")

        activity_detector = ActivityDetector()
        print("✅ ActivityDetector inicializado")

        anomaly_detector = AnomalyDetector()
        print("✅ AnomalyDetector inicializado")

        return True

    except Exception as e:
        print(f"❌ Erro ao inicializar detectores: {e}")
        return False


def test_configuration():
    print("\nTestando configuração...")

    try:
        from src.utils import CONFIG

        print("Configuração atual:")
        for key, value in CONFIG.items():
            print(f"   - {key}: {value}")

        print("✅ Configuração carregada com sucesso")
        return True

    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False


def run_all_tests():
    print("🧪 TESTE DE INSTALAÇÃO - Análise de Vídeo")
    print("=" * 60)

    tests = [
        ("Importações de Dependências", test_imports),
        ("Módulos do Projeto", test_src_modules),
        ("Arquivo de Vídeo", test_video_file),
        ("Diretórios", test_directories),
        ("Modelos de IA", test_models),
        ("Funcionalidade Básica", test_basic_functionality),
        ("Configuração", test_configuration),
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")

        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")

    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: {passed_tests}/{total_tests} testes passaram")

    if passed_tests == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM! A instalação está correta.")
        print("\nPróximos passos:")
        print("1. Execute: python main.py data/video.mp4")
        print("2. Para modo rápido: python main.py data/video.mp4 --fast-mode")
        print("3. Para preview: python main.py data/video.mp4 --show-preview")
        return True
    else:
        print("⚠️  ALGUNS TESTES FALHARAM. Verifique os erros acima.")
        print("\nSugestões:")
        print("1. Instale as dependências: pip install -r requirements.txt")
        print("2. Verifique se o arquivo video.mp4 está em data/")
        print("3. Certifique-se de ter Python 3.8+ instalado")
        return False


def main():
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTeste interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

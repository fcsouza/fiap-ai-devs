# Análise de Vídeo com Reconhecimento Facial

Uma aplicação Python completa para análise de vídeo que incorpora reconhecimento facial, análise de expressões emocionais, detecção de atividades e identificação de anomalias.

## 🚀 Funcionalidades

### ✅ Reconhecimento Facial
- Detecção e rastreamento de rostos usando MediaPipe
- Identificação única de pessoas através de cenas
- Sistema de re-identificação para evitar contagem duplicada
- Rastreamento temporal com IDs únicos

### ✅ Análise de Expressões Emocionais
- Análise de 7 emoções: Feliz, Triste, Raiva, Medo, Surpresa, Nojo, Neutro
- Modelo DeepFace com FER2013
- Cache de emoções para melhor performance
- Análise de mudanças emocionais ao longo do tempo

### ✅ Detecção de Atividades
- Detecção de pessoas usando YOLO
- Análise de pose usando MediaPipe
- Classificação de atividades: Caminhar, Sentar, Ficar em pé, Gesticular, Falar, Escrever, Digitar
- Rastreamento de movimento e velocidade

### ✅ Detecção de Anomalias
- Identificação de movimentos incomuns
- Detecção de comportamentos atípicos
- Análise de padrões de movimento anômalos
- Classificação de anomalias por tipo e severidade

### ✅ Geração de Relatórios
- Relatórios em texto e JSON
- Visualizações gráficas (gráficos de barras, tabelas)
- Estatísticas detalhadas
- Insights automáticos

## 📋 Requisitos

### Sistema
- Python 3.8+
- OpenCV
- CUDA (opcional, para aceleração GPU)

### Dependências Python
```
opencv-python==4.8.1.78
numpy==1.24.3
mediapipe==0.10.7
tensorflow==2.13.0
keras==2.13.1
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
Pillow==10.0.0
deepface==0.0.79
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2
```

## 🛠️ Instalação

### 1. Clone o Repositório
```bash
git clone <url-do-repositorio>
cd tech-challenge04-a
```

### 2. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe os Modelos (Opcional)
Os modelos serão baixados automaticamente na primeira execução:
- YOLOv8n para detecção de pessoas
- Modelos DeepFace para análise emocional

## 🎯 Como Usar

### Uso Básico
```bash
python main.py caminho/para/video.mp4
```

### Modo Rápido (Performance Otimizada)
```bash
python main.py caminho/para/video.mp4 --fast-mode
```

### Com Preview em Tempo Real
```bash
python main.py caminho/para/video.mp4 --show-preview
```

### Diretório de Saída Personalizado
```bash
python main.py caminho/para/video.mp4 --output-dir meus_relatorios
```

## ⚙️ Configuração

### Parâmetros Principais (src/utils.py)
```python
CONFIG = {
    "face_detection_confidence": 0.5,      
    "emotion_confidence_threshold": 0.6,    
    "activity_confidence_threshold": 0.7,   
    "anomaly_threshold": 0.8,              
    "max_faces": 10,                       # Máximo de rostos a detectar
    "frame_skip": 3,                       # Processar a cada N frames
    "max_frame_width": 640,                # Largura máxima dos frames
    "skip_emotion_analysis": False,        # Pular análise emocional
    "skip_anomaly_detection": False,       # Pular detecção de anomalias
}
```

### Modo Rápido
O modo rápido (`--fast-mode`) otimiza a performance:
- Processa a cada 5º frame
- Reduz resolução para 480px
- Desabilita análise emocional
- Desabilita detecção de anomalias
- Desabilita cache de emoções

## 📁 Estrutura do Projeto

```
tech-challenge04-a/
├── main.py                 # Ponto de entrada principal
├── requirements.txt        # Dependências Python
├── README.md              # Este arquivo
├── example.py             # Exemplos de uso
├── test_installation.py   # Script de teste
├── data/                  # Vídeos de entrada
│   └── video.mp4
├── output/                # Relatórios gerados
│   ├── video_analysis_report_*.txt
│   ├── video_analysis_report_*.json
│   └── *.png (gráficos)
├── temp/                  # Arquivos temporários
└── src/                   # Código fonte
    ├── __init__.py
    ├── utils.py           # Utilitários e configuração
    ├── video_processor.py # Processamento de vídeo
    ├── face_detector.py   # Detecção facial
    ├── emotion_analyzer.py # Análise emocional
    ├── activity_detector.py # Detecção de atividades
    ├── anomaly_detector.py # Detecção de anomalias
    └── report_generator.py # Geração de relatórios
```

## 📊 Saída e Relatórios

### Relatórios Gerados
1. **Relatório de Texto** (`*.txt`)
   - Informações do vídeo
   - Estatísticas de detecção
   - Distribuição de emoções e atividades
   - Tipos de anomalias detectadas
   - Insights principais

2. **Relatório JSON** (`*.json`)
   - Dados estruturados
   - Estatísticas detalhadas
   - Metadados da análise

3. **Visualizações** (`*.png`)
   - Gráfico de distribuição de emoções
   - Gráfico de distribuição de atividades
   - Gráfico de tipos de anomalias
   - Tabela de resumo da análise

### Exemplo de Saída
```
VIDEO ANALYSIS SUMMARY
============================================================
Video Duration: 2:15
Frames Analyzed: 3,240
Unique Faces Detected: 1
Unique Emotions Found: 517
Unique Activities Detected: 30
Unique Anomalies Found: 0
Frames with Faces: 715
Frames with Emotions: 715
Frames with Activities: 1,200
Frames with Anomalies: 0
Processing Time: 45.23 seconds
Dominant Emotion: sad
Dominant Activity: sitting
```

## 🔧 Uso Programático

### Exemplo Básico
```python
from main import VideoAnalyzer

analyzer = VideoAnalyzer(
    video_path="data/video.mp4",
    output_dir="output",
    show_preview=False,
    fast_mode=False
)

report = analyzer.analyze_video()
print(f"Análise concluída: {report['statistics']['analysis_summary']}")
```

### Exemplo Personalizado
```python
from src.face_detector import FaceDetector
from src.emotion_analyzer import EmotionAnalyzer
from src.activity_detector import ActivityDetector
from src.anomaly_detector import AnomalyDetector

# Inicializar detectores
face_detector = FaceDetector()
emotion_analyzer = EmotionAnalyzer()
activity_detector = ActivityDetector()
anomaly_detector = AnomalyDetector()

# Processar frame
faces = face_detector.detect_faces(frame)
emotions = emotion_analyzer.analyze_faces_emotions(frame, faces)
activities = activity_detector.detect_activities(frame)
anomalies = anomaly_detector.detect_anomalies(frame, faces, emotions, activities)
```

## 🚀 Performance

### Otimizações Implementadas
- **Frame Skipping**: Processa a cada N frames
- **Redimensionamento**: Limita largura máxima dos frames
- **Cache de Emoções**: Evita re-análise de rostos similares
- **Modo Rápido**: Desabilita análises custosas
- **Processamento Paralelo**: Suporte para múltiplos cores

### Benchmarks Típicos
- **Vídeo 2min, 30fps**: ~45 segundos (modo normal)
- **Vídeo 2min, 30fps**: ~15 segundos (modo rápido)
- **Uso de Memória**: ~2-4GB RAM
- **Uso de CPU**: 80-100% (modo normal), 40-60% (modo rápido)


### Bibliotecas Utilizadas
- **OpenCV**: Processamento de vídeo e visão computacional
- **MediaPipe**: Detecção facial e análise de pose
- **DeepFace**: Análise de expressões emocionais
- **YOLO**: Detecção de objetos e pessoas
- **scikit-learn**: Detecção de anomalias
- **Matplotlib/Seaborn**: Visualizações

### Recursos
- Modelos pré-treinados do MediaPipe
- Dataset FER2013 para emoções
- Arquitetura YOLOv8 para detecção
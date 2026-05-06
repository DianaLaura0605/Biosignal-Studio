# 🧬 BioSignal Studio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.13-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![WebSockets](https://img.shields.io/badge/WebSockets-Real--time-FF6B6B?style=for-the-badge)

**Plataforma de adquisición, procesamiento y análisis de señales biomédicas para control de prótesis biónicas.**

*Diseñada para aplicaciones médicas reales. Construida con los mismos estándares que usan Sensata Technologies, Texas Instruments y Ottobock.*

[Demo en vivo](#) · [Documentación API](#documentación-api) · [Arquitectura](#arquitectura)

</div>

---

## ¿Qué es BioSignal Studio?

BioSignal Studio es una plataforma full-stack para el **procesamiento en tiempo real de señales biomédicas** orientada al control de prótesis biónicas mioeléctricas. Implementa la cadena completa de procesamiento de señales que va desde la adquisición (hardware) hasta la clasificación de gestos musculares.

El sistema procesa tres tipos de señales biomédicas fundamentales:

| Señal | Frecuencia | Amplitud | Aplicación |
|-------|-----------|----------|-----------|
| **EMG** (Electromiografía) | 2000 Hz | 0.1–5 mV | Control de prótesis mioeléctricas |
| **ECG** (Electrocardiografía) | 500 Hz | 0.5–2 mV | Monitoreo cardíaco integrado |
| **EEG** (Electroencefalografía) | 256 Hz | 10–100 µV | Interfaces cerebro-computador (BCI) |

---

## Características Técnicas

### Procesamiento DSP (Digital Signal Processing)
- **Filtro Butterworth** orden 4, zero-phase (filtfilt) — máxima planitud en banda de paso
- **Filtro Notch IIR** a 60 Hz — eliminación de interferencia de red eléctrica
- **Análisis FFT** con ventana Hanning — reduce spectral leakage
- **Extracción de características**: RMS, MAV, ZCR, varianza, potencia espectral por banda

### Clasificación de Gestos EMG
Implementa las 6 clases de gestos estándar para prótesis mioeléctricas:
- Reposo · Puño cerrado · Pinza lateral · Extensión · Pronación · Supinación

Basado en la metodología de Phinyomark et al. (2012) — el paper más citado en control de prótesis.

### Streaming en Tiempo Real
- **WebSockets** con FastAPI — latencia < 5 ms
- Buffer circular de 5000 muestras por canal
- 20 paquetes/segundo, 50 muestras por paquete
- Soporte para múltiples clientes simultáneos

### API REST Completa
- Documentación automática con Swagger UI y ReDoc
- Modelos Pydantic con validación estricta
- Endpoints para snapshots, análisis completo y gestión de sesiones clínicas

---

## Arquitectura

```
biosignal-studio/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app + WebSocket endpoint
│   │   ├── api/
│   │   │   ├── signals.py           # REST endpoints de señales
│   │   │   ├── analysis.py          # Análisis DSP on-demand
│   │   │   ├── prosthetics.py       # Clasificación de gestos
│   │   │   └── sessions.py          # Gestión de sesiones clínicas
│   │   ├── core/
│   │   │   ├── config.py            # Settings con Pydantic BaseSettings
│   │   │   └── connection_manager.py # Gestor WebSocket multi-cliente
│   │   └── services/
│   │       ├── signal_generator.py  # Generador de señales biomédicas
│   │       └── dsp_processor.py     # Pipeline DSP (filtros + características)
│   └── requirements.txt
└── frontend/
    └── index.html                   # SPA con Canvas 2D + WebSocket client
```

### Stack tecnológico

**Backend:**
- **FastAPI** — framework async de alto rendimiento con documentación automática
- **NumPy** — operaciones vectoriales sobre arrays de señal (sin loops Python)
- **SciPy** — filtros IIR/FIR profesionales (`scipy.signal`)
- **WebSockets** — streaming bidireccional en tiempo real
- **Pydantic** — validación de datos y serialización tipada

**Frontend:**
- **HTML5 Canvas** — osciloscopio en tiempo real a 60 fps
- **WebSocket API** — recepción de streaming de señal
- **Vanilla JS** — sin dependencias, máximo rendimiento

---

## Instalación y uso

### Requisitos
- Python 3.11+
- pip

### Backend

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/biosignal-studio.git
cd biosignal-studio/backend

# Entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dependencias
pip install -r requirements.txt

# Iniciar servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
# Abrir directamente en el navegador
open ../frontend/index.html

# O servir con Python
cd ../frontend
python -m http.server 3000
```

### Documentación API
Una vez iniciado el servidor:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## Documentación API

### Clasificación de gesto (POST /api/prosthetics/classify)
```json
{
  "emg_window": [0.12, -0.08, 0.23, "..."],
  "sample_rate": 2000,
  "channel_count": 1
}
```
**Respuesta:**
```json
{
  "gesture_id": 1,
  "gesture_name": "Puño cerrado",
  "confidence": 0.847,
  "latency_ms": 2.341,
  "features": {
    "rms": 0.342,
    "mav": 0.271,
    "zero_crossings": 87,
    "dominant_freq": 112.4
  }
}
```

### WebSocket — Streaming
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/cliente-001');

// Iniciar stream EMG
ws.send(JSON.stringify({ command: 'start_stream', signal_type: 'emg' }));

// Recibir paquetes de datos
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.raw: señal cruda (Float64[])
  // data.processed: señal filtrada (Float64[])
  // data.rms: nivel de activación muscular
  // data.dominant_freq: frecuencia dominante (Hz)
};
```

---

## Relevancia Industrial

Este proyecto implementa tecnologías activamente usadas en la industria:

- **Sensata Technologies**: Sus sensores de presión y corriente son el hardware de adquisición en sistemas EMG médicos. Este software procesaría sus señales de salida.
- **Texas Instruments**: El ADS1299 (amplificador AFE para bio-señales de TI) es el ADC más usado en equipos EEG/ECG. Este backend es compatible con su output.
- **Micron Technology**: Los sistemas de prótesis biónicas avanzadas usan memoria LPDDR4/5 de Micron en el procesador embebido. La arquitectura de buffer circular de este proyecto está optimizada para memoria limitada.
- **Intel**: Intel OpenVINO puede acelerar el clasificador de gestos con inferencia en hardware dedicado.

---

## Fundamento Científico

### Modelo EMG
Basado en el modelo estadístico de **Basmajian & De Luca (1985)**: la señal EMG superficial se modela como la superposición estocástica de Potenciales de Acción de Unidades Motoras (MUAPs).

### Clasificación de gestos
Implementa las características de **Phinyomark, Phukpattaranont & Limsakul (2012)**: *"Feature reduction and selection for EMG signal classification"* — el survey más citado en el área (2000+ citas).

### Procesamiento en tiempo real
Arquitectura inspirada en el sistema de control de la **DEKA LUKE Arm** — la prótesis de mano más avanzada desarrollada para DARPA con latencia < 20ms.

---

## Extensiones futuras

- [ ] Integración con hardware real (OpenBCI, ADS1299 de TI, BioAmp EXG Pill)
- [ ] Clasificador ML entrenado (SVM / Random Forest / LSTM) con scikit-learn
- [ ] Base de datos PostgreSQL + SQLAlchemy para sesiones clínicas
- [ ] Soporte multi-canal (8 canales EMG simultáneos)
- [ ] Dashboard de telemetría para fisioterapeutas
- [ ] Exportación a formato EDF (European Data Format) estándar médico
- [ ] Docker + CI/CD con GitHub Actions

---

## Autora Diana Laura Bocardo

**Ingeniera Biomédica** · Especialización en Prótesis Biónicas y Diseño 3D

*Construido con amor por la ingeniería biomédica y el procesamiento de señales.*

---

<div align="center">
<sub>BioSignal Studio · MIT License · Hecho en México 🇲🇽</sub>
</div>


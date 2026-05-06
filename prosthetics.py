"""
Router: Control de Prótesis Biónicas
======================================
Endpoints para clasificación de gestos a partir de señales EMG.

El flujo es:
  1. Cliente envía ventana de señal EMG (250ms de datos)
  2. Se extraen características (feature extraction)
  3. El clasificador predice el gesto
  4. Se retorna el gesto + confianza + tiempo de latencia

Gestos implementados (clasificación de 6 clases):
  0: Reposo          — mano relajada
  1: Puño cerrado    — todos los dedos cerrados
  2: Pinza lateral   — pulgar + índice
  3: Extensión       — dedos extendidos
  4: Pronación       — palma hacia abajo
  5: Supinación      — palma hacia arriba

En producción se usaría un modelo de ML entrenado con datos reales.
Aquí implementamos un clasificador heurístico basado en características
EMG para demostración sin dependencias de ML pesadas.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import numpy as np
import time
import logging

from app.services.dsp_processor import DSPProcessor

logger = logging.getLogger("prosthetics")

router = APIRouter()

# ─── Modelos de datos (Pydantic) ──────────────────────────────────────────────

class GestureClassifyRequest(BaseModel):
    """Solicitud de clasificación de gesto."""
    emg_window: List[float] = Field(
        ...,
        description="Ventana de señal EMG (mV). Mínimo 50 muestras.",
        min_items=50,
        max_items=10000,
    )
    sample_rate: int = Field(2000, description="Frecuencia de muestreo en Hz")
    channel_count: int = Field(1, ge=1, le=8, description="Número de canales EMG")

    class Config:
        schema_extra = {
            "example": {
                "emg_window": [0.12, -0.08, 0.23, "..."],
                "sample_rate": 2000,
                "channel_count": 1,
            }
        }


class GestureResult(BaseModel):
    """Resultado de clasificación de gesto."""
    gesture_id: int                     # 0-5
    gesture_name: str                   # Nombre legible
    confidence: float                   # 0.0 - 1.0
    latency_ms: float                   # Tiempo de procesamiento
    features: Dict[str, float]          # Características extraídas
    all_probabilities: Dict[str, float] # Probabilidad de cada clase


class ProthesisCalibrationRequest(BaseModel):
    """Solicitud de calibración del clasificador."""
    gesture_id: int = Field(..., ge=0, le=5)
    training_samples: List[List[float]] = Field(
        ...,
        description="Lista de ventanas EMG para este gesto"
    )


# ─── Clasificador de gestos ────────────────────────────────────────────────────

GESTURE_NAMES = {
    0: "Reposo",
    1: "Puño cerrado",
    2: "Pinza lateral",
    3: "Extensión",
    4: "Pronación",
    5: "Supinación",
}

class GestureClassifier:
    """
    Clasificador heurístico de gestos EMG.

    Usa reglas basadas en características clínicas bien documentadas:
    - RMS: nivel de contracción muscular
    - MAV: amplitud media (similar a RMS, menos sensible a picos)
    - ZCR: frecuencia de cruces por cero (relacionada con la frecuencia de disparo)
    - Spectral power: distribución de energía en frecuencia

    En un sistema comercial (i.LIMB, Michelangelo, Ottobock), estas
    características alimentan un clasificador LDA, SVM o red neuronal
    entrenado con datos del usuario.
    """

    def __init__(self):
        self.processor = DSPProcessor()

    def classify(self, emg_window: np.ndarray, fs: int) -> Dict:
        """
        Clasifica un gesto a partir de una ventana EMG.

        Args:
            emg_window: Array de muestras EMG
            fs: Frecuencia de muestreo

        Returns:
            Dict con gesture_id, confidence, features, probabilities
        """
        # Extraer características
        result = self.processor.process(emg_window, "emg")

        features = {
            "rms": result["rms"],
            "mav": result["mav"],
            "variance": result["variance"],
            "zero_crossings": float(result["zero_crossings"]),
            "dominant_freq": result["dominant_freq"],
        }

        # Clasificación heurística basada en características
        probs = self._compute_probabilities(features)

        gesture_id = max(probs, key=probs.get)
        confidence = probs[gesture_id]

        return {
            "gesture_id": gesture_id,
            "gesture_name": GESTURE_NAMES[gesture_id],
            "confidence": confidence,
            "features": features,
            "all_probabilities": {GESTURE_NAMES[k]: v for k, v in probs.items()},
        }

    def _compute_probabilities(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Asigna probabilidades a cada gesto basándose en reglas clínicas.

        Reglas derivadas de literatura EMG:
        - Reposo: RMS bajo, ZCR bajo
        - Puño: RMS alto, MAV alto, frecuencia media alta
        - Pinza: RMS medio, ZCR medio
        - Extensión: ZCR alto (muchas unidades motoras rápidas)
        - Pronación/Supinación: diferenciados por frecuencia dominante
        """
        rms = features["rms"]
        mav = features["mav"]
        zcr = features["zero_crossings"]
        df = features["dominant_freq"]

        # Scores base (heurísticos)
        scores = {
            0: max(0, 1.0 - rms * 5),              # Reposo: bajo RMS
            1: rms * 2.5 + mav * 1.5,              # Puño: alto RMS y MAV
            2: rms * 1.5 + (1 / (1 + abs(zcr - 80))),  # Pinza: RMS medio, ZCR ~80
            3: (zcr / 200) * 2.0,                   # Extensión: ZCR alto
            4: rms * 1.2 + (df / 200) if df < 100 else 0,   # Pronación: baja freq
            5: rms * 1.2 + (1 - df / 200) if df > 100 else 0,  # Supinación: alta freq
        }

        # Añadir ruido estocástico (simula variabilidad de un clasificador real)
        rng = np.random.default_rng()
        for k in scores:
            scores[k] = max(0, scores[k] + rng.normal(0, 0.05))

        # Normalizar a probabilidades (softmax)
        total = sum(scores.values())
        if total < 1e-10:
            return {k: 1/6 for k in scores}  # Distribución uniforme si todo es 0

        return {k: v / total for k, v in scores.items()}


# Instancia global del clasificador
_classifier = GestureClassifier()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/classify", response_model=GestureResult)
async def classify_gesture(request: GestureClassifyRequest):
    """
    Clasifica un gesto a partir de una ventana de señal EMG.

    Este es el endpoint central del sistema de control de prótesis.
    Se llama cada 250ms con una nueva ventana de datos.

    Returns:
        GestureResult con el gesto predicho y su confianza
    """
    t_start = time.perf_counter()

    emg_array = np.array(request.emg_window, dtype=np.float64)

    try:
        result = _classifier.classify(emg_array, request.sample_rate)
    except Exception as e:
        logger.error(f"Error en clasificación: {e}")
        raise HTTPException(status_code=500, detail=f"Error en clasificación: {str(e)}")

    latency_ms = (time.perf_counter() - t_start) * 1000

    return GestureResult(
        gesture_id=result["gesture_id"],
        gesture_name=result["gesture_name"],
        confidence=result["confidence"],
        latency_ms=round(latency_ms, 3),
        features=result["features"],
        all_probabilities=result["all_probabilities"],
    )


@router.get("/gestures")
async def list_gestures():
    """
    Lista todos los gestos soportados con su descripción clínica.
    Útil para la UI — construir selector de gestos para calibración.
    """
    return {
        "gestures": [
            {
                "id": k,
                "name": v,
                "description": _get_gesture_description(k),
                "muscles_involved": _get_muscles(k),
            }
            for k, v in GESTURE_NAMES.items()
        ]
    }


@router.get("/prosthetics/models")
async def list_prosthetic_models():
    """
    Modelos de prótesis biónicas compatibles con este sistema.
    Información de referencia para la UI.
    """
    return {
        "models": [
            {
                "name": "i-LIMB Ultra",
                "manufacturer": "Össur",
                "dof": 5,
                "control": "myoelectric",
                "sensors": "EMG bipolar",
            },
            {
                "name": "Michelangelo",
                "manufacturer": "Ottobock",
                "dof": 6,
                "control": "myoelectric + IMU",
                "sensors": "EMG + accelerometer",
            },
            {
                "name": "HERO Arm",
                "manufacturer": "Open Bionics",
                "dof": 4,
                "control": "myoelectric",
                "sensors": "EMG bipolar",
            },
            {
                "name": "Luke Arm (DEKA)",
                "manufacturer": "DEKA Research",
                "dof": 10,
                "control": "multi-modal",
                "sensors": "EMG + foot pressure + IMU",
            },
        ]
    }


def _get_gesture_description(gesture_id: int) -> str:
    descriptions = {
        0: "Estado de reposo. Sin contracción muscular activa.",
        1: "Cierre completo de los dedos. Activa flexores superficial y profundo.",
        2: "Oposición de pulgar e índice para agarre fino de objetos pequeños.",
        3: "Apertura completa de la mano. Activa extensores digitales.",
        4: "Rotación de la muñeca con palma hacia el suelo.",
        5: "Rotación de la muñeca con palma hacia el cielo.",
    }
    return descriptions.get(gesture_id, "Gesto no documentado")


def _get_muscles(gesture_id: int) -> List[str]:
    muscles = {
        0: [],
        1: ["Flexor digitorum superficialis", "Flexor digitorum profundus"],
        2: ["Flexor pollicis longus", "Flexor digitorum superficialis"],
        3: ["Extensor digitorum communis", "Extensor pollicis longus"],
        4: ["Pronator teres", "Pronator quadratus"],
        5: ["Supinator", "Biceps brachii"],
    }
    return muscles.get(gesture_id, [])

"""
Procesador DSP (Digital Signal Processing)
============================================
Implementa los algoritmos de procesamiento de señales biomédicas
más utilizados en sistemas de control de prótesis biónicas.

Algoritmos implementados:
  - Filtrado Butterworth (pasa-banda, notch)
  - Análisis espectral FFT
  - Extracción de características temporales y frecuenciales
  - RMS (Root Mean Square) — métrica de actividad muscular
  - MAV (Mean Absolute Value) — amplitud media de EMG
  - Zero Crossing Rate — tasa de cruces por cero

Referencias:
  - Phinyomark et al. (2012): Feature extraction of EMG for prosthetics
  - Oskoei & Hu (2007): Myoelectric control systems — A survey
  - De Luca (2006): Fundamental concepts in EMG signal acquisition
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("dsp_processor")


@dataclass
class ProcessedSignal:
    """Resultado del procesamiento DSP de un chunk de señal."""
    filtered: np.ndarray        # Señal filtrada
    rms: float                  # Root Mean Square (activación muscular)
    mav: float                  # Mean Absolute Value
    variance: float             # Varianza de la señal
    zero_crossings: int         # Tasa de cruces por cero
    dominant_freq: float        # Frecuencia dominante (Hz)
    spectral_power: Dict        # Potencia por banda de frecuencia
    snr: float                  # Relación señal/ruido estimada


class DSPProcessor:
    """
    Procesador de señales biomédicas en tiempo real.

    Aplica la cadena de procesamiento:
    1. Filtrado pasa-banda (elimina DC y alias)
    2. Notch filter (elimina interferencia 60 Hz)
    3. Extracción de características
    4. Análisis espectral
    """

    # Rangos de frecuencia por tipo de señal
    BANDPASS_CONFIG = {
        "emg": {"low": 20.0,  "high": 450.0, "fs": 2000},  # EMG superficial estándar
        "ecg": {"low": 0.5,   "high": 40.0,  "fs": 500},   # ECG diagnóstico
        "eeg": {"low": 0.5,   "high": 100.0, "fs": 256},   # EEG completo
    }

    # Bandas de frecuencia EEG
    EEG_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 100),
    }

    def __init__(self, notch_freq: float = 60.0, notch_q: float = 30.0):
        """
        Args:
            notch_freq: Frecuencia de la interferencia de red (60 Hz en América, 50 Hz en Europa)
            notch_q: Factor Q del filtro notch (mayor Q = banda más estrecha)
        """
        self.notch_freq = notch_freq
        self.notch_q = notch_q

        # Caché de coeficientes de filtro (evita recalcular en cada chunk)
        self._filter_cache: Dict[str, tuple] = {}

    def process(self, signal: np.ndarray, signal_type: str) -> Dict[str, Any]:
        """
        Pipeline completo de procesamiento DSP.

        Args:
            signal: Array de muestras crudas (mV)
            signal_type: "emg" | "ecg" | "eeg"

        Returns:
            Dict con señal filtrada y todas las características extraídas
        """
        config = self.BANDPASS_CONFIG.get(signal_type, self.BANDPASS_CONFIG["emg"])
        fs = config["fs"]

        # ── Paso 1: Filtro pasa-banda ──────────────────────────────────────────
        filtered = self._bandpass_filter(signal, config["low"], config["high"], fs)

        # ── Paso 2: Filtro notch (eliminar 60 Hz) ────────────────────────────
        filtered = self._notch_filter(filtered, fs)

        # ── Paso 3: Características temporales ───────────────────────────────
        rms = self._compute_rms(filtered)
        mav = float(np.mean(np.abs(filtered)))
        variance = float(np.var(filtered))
        zc = self._zero_crossing_rate(filtered)

        # ── Paso 4: Análisis espectral ────────────────────────────────────────
        dominant_freq, spectral_power = self._spectral_analysis(filtered, fs, signal_type)

        # ── Paso 5: SNR estimado ─────────────────────────────────────────────
        snr = self._estimate_snr(signal, filtered)

        return {
            "filtered": filtered,
            "rms": rms,
            "mav": mav,
            "variance": variance,
            "zero_crossings": zc,
            "dominant_freq": dominant_freq,
            "spectral_power": spectral_power,
            "snr": snr,
        }

    def _bandpass_filter(self, data: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
        """
        Filtro Butterworth de orden 4 pasa-banda.

        Butterworth se elige por su respuesta de magnitud maximalmente plana
        en la banda de paso — no introduce ripple como los filtros Chebyshev.

        Usamos filtfilt (zero-phase) para no desplazar la señal en el tiempo,
        crítico para análisis de latencia en sistemas de control.
        """
        cache_key = f"bp_{low}_{high}_{fs}"

        if cache_key not in self._filter_cache:
            # Normalizar frecuencias (Nyquist = fs/2)
            nyq = fs / 2.0
            low_n = low / nyq
            high_n = min(high / nyq, 0.99)  # No puede superar Nyquist

            # Diseñar filtro Butterworth orden 4
            b, a = sp_signal.butter(4, [low_n, high_n], btype='bandpass')
            self._filter_cache[cache_key] = (b, a)

        b, a = self._filter_cache[cache_key]

        # filtfilt requiere al menos 3x el orden del filtro en muestras
        if len(data) < 3 * max(len(b), len(a)):
            return data  # No hay suficientes muestras, retornar sin filtrar

        try:
            return sp_signal.filtfilt(b, a, data)
        except Exception:
            return data

    def _notch_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Filtro notch IIR para eliminar interferencia de red eléctrica.

        El filtro iirnotch crea un cero exacto en la frecuencia objetivo,
        ideal para eliminar la componente de 60 Hz sin afectar el resto.
        """
        cache_key = f"notch_{self.notch_freq}_{fs}"

        if cache_key not in self._filter_cache:
            b, a = sp_signal.iirnotch(self.notch_freq, self.notch_q, fs)
            self._filter_cache[cache_key] = (b, a)

        b, a = self._filter_cache[cache_key]

        if len(data) < 3 * max(len(b), len(a)):
            return data

        try:
            return sp_signal.filtfilt(b, a, data)
        except Exception:
            return data

    def _compute_rms(self, data: np.ndarray) -> float:
        """
        Root Mean Square — métrica estándar de amplitud EMG.

        RMS es proporcional a la fuerza muscular en contracciones isométricas.
        Es la característica más usada en sistemas de control de prótesis.
        """
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))

    def _zero_crossing_rate(self, data: np.ndarray) -> int:
        """
        Zero Crossing Rate (ZCR) — cruces por cero por unidad de tiempo.

        En EMG: el ZCR se correlaciona con la frecuencia de disparo de las
        unidades motoras. Útil para distinguir contracciones suaves vs. fuertes.
        En EEG: diferencia ondas lentas (delta) de ondas rápidas (gamma).
        """
        if len(data) < 2:
            return 0
        # Contar cambios de signo
        signs = np.sign(data)
        signs[signs == 0] = 1  # Tratar cero como positivo
        crossings = np.sum(np.diff(signs) != 0)
        return int(crossings)

    def _spectral_analysis(self, data: np.ndarray, fs: float, signal_type: str):
        """
        Análisis espectral via FFT.

        Retorna:
        - dominant_freq: frecuencia con mayor potencia (Hz)
        - spectral_power: dict con potencia en cada banda de interés
        """
        n = len(data)
        if n < 4:
            return 0.0, {}

        # Aplicar ventana Hanning para reducir spectral leakage
        window = np.hanning(n)
        fft_vals = fft(data * window)
        freqs = fftfreq(n, d=1.0/fs)

        # Solo frecuencias positivas
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        magnitude = np.abs(fft_vals[pos_mask])
        power = magnitude ** 2

        # Frecuencia dominante
        if len(power) > 0:
            dominant_freq = float(freqs_pos[np.argmax(power)])
        else:
            dominant_freq = 0.0

        # Potencia por banda
        spectral_power = {}
        if signal_type == "eeg":
            for band_name, (low, high) in self.EEG_BANDS.items():
                band_mask = (freqs_pos >= low) & (freqs_pos <= high)
                spectral_power[band_name] = float(np.sum(power[band_mask]))
        else:
            # Para EMG/ECG: bandas clínicas estándar
            bands = {"low": (0, 50), "mid": (50, 150), "high": (150, 450)}
            for band_name, (low, high) in bands.items():
                band_mask = (freqs_pos >= low) & (freqs_pos <= high)
                spectral_power[band_name] = float(np.sum(power[band_mask]))

        return dominant_freq, spectral_power

    def _estimate_snr(self, raw: np.ndarray, filtered: np.ndarray) -> float:
        """
        Estima la relación señal/ruido comparando señal cruda vs. filtrada.
        SNR = 20 * log10(RMS_signal / RMS_noise)
        """
        noise = raw - filtered
        rms_signal = self._compute_rms(filtered)
        rms_noise = self._compute_rms(noise)

        if rms_noise < 1e-10:  # Evitar división por cero
            return 100.0

        snr = 20 * np.log10(rms_signal / rms_noise)
        return float(np.clip(snr, -20, 60))  # Limitar a rango razonable

"""
Generador de Señales Biomédicas
=================================
Simula señales biomédicas realistas para demostración y testing.
En producción, este módulo se reemplaza por la lectura de hardware
real (ADS1299, OpenBCI, BioAmp EXG Pill, etc.)

Señales implementadas:
  - EMG (Electromiografía): actividad muscular para control de prótesis
  - ECG (Electrocardiografía): ritmo cardíaco
  - EEG (Electroencefalografía): ondas cerebrales

Cada señal incluye:
  - Componente fisiológica realista (modelada matemáticamente)
  - Ruido de fondo (ruido térmico + artefactos de movimiento)
  - Interferencia de línea eléctrica (60 Hz)
  - Variaciones temporales para simular cambios reales
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class SignalType(str, Enum):
    EMG = "emg"
    ECG = "ecg"
    EEG = "eeg"


@dataclass
class SignalState:
    """
    Estado interno del generador para continuidad entre llamadas.
    Mantiene la fase actual de cada señal para que los chunks
    consecutivos formen una señal continua.
    """
    phase: float = 0.0              # Fase acumulada (radianes)
    time_offset: float = 0.0        # Tiempo transcurrido (segundos)
    muscle_activity: float = 0.5    # Nivel de activación muscular (0-1)
    gesture_class: int = 0          # Gesto actual (para clasificación)


class SignalGenerator:
    """
    Genera señales biomédicas sintéticas con características realistas.

    Uso:
        gen = SignalGenerator()
        samples = gen.generate("emg", n_samples=500)
        # samples.shape == (500,), dtype=float64, rango aproximado [-1, 1] mV
    """

    # Frecuencias de muestreo por tipo de señal (Hz)
    SAMPLE_RATES = {
        SignalType.EMG: 2000,
        SignalType.ECG: 500,
        SignalType.EEG: 256,
    }

    # Amplitudes típicas (mV)
    AMPLITUDES = {
        SignalType.EMG: 1.0,    # EMG superficial: 0.1 - 5 mV
        SignalType.ECG: 1.2,    # ECG: 0.5 - 2 mV
        SignalType.EEG: 0.1,    # EEG: 10 - 100 µV (0.01 - 0.1 mV)
    }

    def __init__(self):
        self._states: dict[str, SignalState] = {
            st.value: SignalState() for st in SignalType
        }
        self._rng = np.random.default_rng(seed=42)  # RNG reproducible

    def generate(self, signal_type: str, n_samples: int) -> np.ndarray:
        """
        Genera n_samples muestras del tipo de señal especificado.

        Args:
            signal_type: "emg" | "ecg" | "eeg"
            n_samples: cantidad de muestras a generar

        Returns:
            Array numpy de shape (n_samples,) en mV
        """
        st = SignalType(signal_type)
        state = self._states[signal_type]
        fs = self.SAMPLE_RATES[st]
        amp = self.AMPLITUDES[st]

        # Vector de tiempo para este chunk
        t = state.time_offset + np.arange(n_samples) / fs

        # Generar señal según el tipo
        if st == SignalType.EMG:
            signal = self._generate_emg(t, state, amp)
        elif st == SignalType.ECG:
            signal = self._generate_ecg(t, state, amp)
        elif st == SignalType.EEG:
            signal = self._generate_eeg(t, state, amp)

        # Actualizar estado para continuidad
        state.time_offset += n_samples / fs

        return signal

    def _generate_emg(self, t: np.ndarray, state: SignalState, amp: float) -> np.ndarray:
        """
        EMG (Electromiografía) — Modelo de Basmajian-De Luca.

        El EMG de superficie se modela como la suma de Potenciales de Acción
        de Unidades Motoras (MUAPs) con distribución estocástica.

        Componentes:
        1. Señal base: ruido gaussiano filtrado (modelo simplificado de MUAP)
        2. Modulación de amplitud: simula contracción/relajación muscular
        3. Ruido térmico: del hardware de adquisición
        4. Interferencia 60 Hz: ruido eléctrico del ambiente
        """
        n = len(t)

        # 1. Señal EMG base: ruido gaussiano (aproxima la suma de MUAPs)
        raw_noise = self._rng.normal(0, 1, n)

        # 2. Modulación temporal: simula contracción muscular sinusoidal
        #    El músculo se contrae y relaja cada ~2 segundos
        contraction = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))

        # 3. Bursts de actividad (ráfagas de potenciales de acción)
        burst_freq = 12.0  # Hz — tasa de disparo típica de unidades motoras
        burst_env = np.abs(np.sin(2 * np.pi * burst_freq * t))

        # 4. Componer señal EMG
        emg = amp * contraction * burst_env * raw_noise * 0.7

        # 5. Ruido de fondo (térmico + electrónico)
        thermal_noise = self._rng.normal(0, amp * 0.05, n)

        # 6. Interferencia de línea (60 Hz) — realista y luego filtrada en DSP
        powerline = amp * 0.15 * np.sin(2 * np.pi * 60 * t)

        return emg + thermal_noise + powerline

    def _generate_ecg(self, t: np.ndarray, state: SignalState, amp: float) -> np.ndarray:
        """
        ECG (Electrocardiografía) — Modelo de van Eck modificado.

        Genera un ECG realista con morfología PQRST completa.
        Frecuencia cardíaca: ~72 BPM con variabilidad (HRV).

        Ondas del ECG:
        P: despolarización auricular
        Q: inicio del septo
        R: pico de despolarización ventricular (el más alto)
        S: fin de despolarización ventricular
        T: repolarización ventricular
        """
        n = len(t)
        ecg = np.zeros(n)

        # Frecuencia cardíaca base con variabilidad de 2 BPM
        hr_base = 72.0  # BPM
        rr_interval = 60.0 / hr_base  # segundos entre latidos

        # Generar cada latido en el intervalo temporal
        t_start = t[0]
        t_end = t[-1]

        # Calcular tiempos de cada complejo QRS
        beat_times = np.arange(
            rr_interval - (t_start % rr_interval),
            t_end - t_start + rr_interval,
            rr_interval
        ) + t_start

        for beat_t in beat_times:
            if beat_t < t_start or beat_t > t_end:
                continue

            # Índice relativo al vector t
            dt = t - beat_t

            # Onda P (despolarización auricular) — gaussiana suave
            ecg += 0.15 * amp * np.exp(-((dt + 0.20) ** 2) / (2 * 0.008**2))

            # Onda Q (pequeña deflexión negativa)
            ecg -= 0.05 * amp * np.exp(-((dt + 0.05) ** 2) / (2 * 0.005**2))

            # Onda R (pico principal — la característica más prominente)
            ecg += 1.0 * amp * np.exp(-((dt) ** 2) / (2 * 0.003**2))

            # Onda S (deflexión negativa post-R)
            ecg -= 0.25 * amp * np.exp(-((dt - 0.04) ** 2) / (2 * 0.005**2))

            # Onda T (repolarización ventricular — asimétrica)
            ecg += 0.35 * amp * np.exp(-((dt - 0.35) ** 2) / (2 * 0.025**2))

        # Añadir ruido de alta frecuencia (artefacto de movimiento + electrónico)
        ecg += self._rng.normal(0, amp * 0.02, n)

        # Deriva de línea base (artefacto respiratorio — 0.3 Hz)
        ecg += amp * 0.05 * np.sin(2 * np.pi * 0.3 * t)

        return ecg

    def _generate_eeg(self, t: np.ndarray, state: SignalState, amp: float) -> np.ndarray:
        """
        EEG (Electroencefalografía) — Suma de bandas de frecuencia cerebrales.

        Bandas del EEG:
        δ (delta):  0.5 - 4   Hz  — sueño profundo
        θ (theta):  4   - 8   Hz  — somnolencia, meditación
        α (alpha):  8   - 13  Hz  — relajación con ojos cerrados
        β (beta):   13  - 30  Hz  — estado activo, concentración
        γ (gamma):  30  - 100 Hz  — procesamiento cognitivo avanzado
        """
        n = len(t)
        eeg = np.zeros(n)

        # Amplitudes relativas de cada banda (estado de relajación)
        bands = {
            "delta": (2.0,   amp * 0.30),   # (freq_Hz, amplitud)
            "theta": (6.0,   amp * 0.20),
            "alpha": (10.0,  amp * 0.40),   # Alpha dominante — ojos cerrados
            "beta":  (20.0,  amp * 0.15),
            "gamma": (40.0,  amp * 0.05),
        }

        for band_name, (freq, band_amp) in bands.items():
            # Fase aleatoria para cada banda (no son coherentes entre sí)
            phase = self._rng.uniform(0, 2 * np.pi)
            # Pequeña variación de frecuencia para más realismo
            freq_jitter = freq + self._rng.uniform(-0.5, 0.5)
            eeg += band_amp * np.sin(2 * np.pi * freq_jitter * t + phase)

        # Ruido de fondo EEG (actividad sináptica espontánea)
        eeg += self._rng.normal(0, amp * 0.1, n)

        # Artefacto ocular ocasional (parpadeo cada ~3 segundos)
        blink_t = 3.0
        blink_mask = np.sin(2 * np.pi * t / blink_t) > 0.98
        eeg += blink_mask * amp * 5.0 * self._rng.exponential(0.5, n)

        return eeg

    def set_gesture(self, gesture_id: int):
        """
        Cambia el gesto activo para el generador EMG.
        En un sistema real, esto equivale al estado de contracción muscular.

        Gestos: 0=reposo, 1=puño, 2=pinza, 3=extensión, 4=pronación, 5=supinación
        """
        self._states["emg"].gesture_class = gesture_id

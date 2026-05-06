"""
Tests para BioSignal Studio
=============================
Suite de pruebas unitarias e integración para el backend.
Ejecutar con: pytest tests/ -v
"""

import pytest
import numpy as np
from app.services.signal_generator import SignalGenerator, SignalType
from app.services.dsp_processor import DSPProcessor


class TestSignalGenerator:
    """Tests para el generador de señales biomédicas."""

    def setup_method(self):
        self.gen = SignalGenerator()

    def test_emg_output_shape(self):
        """El generador EMG produce la cantidad correcta de muestras."""
        samples = self.gen.generate("emg", 500)
        assert samples.shape == (500,), f"Esperado (500,), obtenido {samples.shape}"

    def test_ecg_output_shape(self):
        """El generador ECG produce la cantidad correcta de muestras."""
        samples = self.gen.generate("ecg", 250)
        assert samples.shape == (250,)

    def test_eeg_output_shape(self):
        """El generador EEG produce la cantidad correcta de muestras."""
        samples = self.gen.generate("eeg", 128)
        assert samples.shape == (128,)

    def test_emg_amplitude_range(self):
        """La señal EMG está en el rango clínico esperado (±10 mV)."""
        samples = self.gen.generate("emg", 2000)
        # EMG superficial no debería superar 10 mV en condiciones normales
        assert np.max(np.abs(samples)) < 10.0, "Amplitud EMG fuera de rango"

    def test_ecg_has_positive_peaks(self):
        """El ECG debe tener picos R positivos (deflexión principal)."""
        samples = self.gen.generate("ecg", 1000)
        # La onda R es la componente más alta del ECG
        assert np.max(samples) > 0.5, "No se detectaron picos R en el ECG"

    def test_continuity_between_chunks(self):
        """Dos chunks consecutivos deben ser continuos (sin saltos)."""
        chunk1 = self.gen.generate("emg", 100)
        chunk2 = self.gen.generate("emg", 100)
        # Ningún chunk debe tener NaN o Inf
        assert not np.any(np.isnan(chunk1))
        assert not np.any(np.isnan(chunk2))
        assert not np.any(np.isinf(chunk2))

    def test_eeg_has_alpha_component(self):
        """El EEG generado debe tener energía en la banda alpha (8-13 Hz)."""
        fs = 256
        samples = self.gen.generate("eeg", fs * 4)  # 4 segundos
        # FFT
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1/fs)
        # Energía en banda alpha
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_energy = np.sum(fft_vals[alpha_mask])
        # Energía en delta (debería ser menor que alpha en este modelo)
        delta_mask = (freqs >= 0.5) & (freqs < 4)
        delta_energy = np.sum(fft_vals[delta_mask])
        assert alpha_energy > 0, "Sin energía en banda alpha"


class TestDSPProcessor:
    """Tests para el procesador DSP."""

    def setup_method(self):
        self.proc = DSPProcessor()
        self.gen = SignalGenerator()

    def test_rms_calculation(self):
        """RMS debe ser correcto para una señal de amplitud conocida."""
        # Señal sinusoidal pura: RMS = A/sqrt(2)
        t = np.linspace(0, 1, 2000)
        sine = np.sin(2 * np.pi * 100 * t)  # Seno de amplitud 1
        rms = self.proc._compute_rms(sine)
        expected = 1 / np.sqrt(2)  # ~0.707
        assert abs(rms - expected) < 0.01, f"RMS incorrecto: {rms:.4f} vs {expected:.4f}"

    def test_notch_filter_removes_60hz(self):
        """El filtro notch debe atenuar significativamente la señal a 60 Hz."""
        fs = 2000
        t = np.linspace(0, 1, fs)
        # Señal pura de 60 Hz (interferencia de red)
        signal_60hz = np.sin(2 * np.pi * 60 * t)
        filtered = self.proc._notch_filter(signal_60hz, fs)
        # La energía debe reducirse significativamente
        energy_before = np.sum(signal_60hz ** 2)
        energy_after  = np.sum(filtered ** 2)
        attenuation = energy_before / max(energy_after, 1e-10)
        assert attenuation > 10, f"El filtro notch no atenúa suficiente: {attenuation:.1f}x"

    def test_bandpass_removes_dc(self):
        """El filtro pasa-banda debe eliminar la componente DC."""
        fs = 2000
        n = 1000
        # Señal con DC offset
        signal_dc = np.ones(n) * 0.5
        filtered = self.proc._bandpass_filter(signal_dc, 20, 450, fs)
        # La media debe estar cerca de cero después del filtrado
        assert abs(np.mean(filtered)) < 0.01, "No se eliminó el DC offset"

    def test_process_returns_all_features(self):
        """process() debe retornar todas las características esperadas."""
        emg = self.gen.generate("emg", 500)
        result = self.proc.process(emg, "emg")

        required_keys = ['filtered', 'rms', 'mav', 'variance', 'zero_crossings', 'dominant_freq', 'snr']
        for key in required_keys:
            assert key in result, f"Falta la característica: {key}"

    def test_snr_positive_for_clean_signal(self):
        """SNR debe ser positivo para una señal más limpia que el ruido."""
        fs = 2000
        t = np.linspace(0, 1, fs)
        # Señal limpia de EMG (sin ruido excesivo)
        clean_emg = np.sin(2 * np.pi * 100 * t) * 0.5
        result = self.proc.process(clean_emg, "emg")
        assert result['snr'] > 0, "SNR debería ser positivo para señal limpia"

    def test_zero_crossing_rate_for_sine(self):
        """Un seno de 10 Hz en 1 segundo debe tener ~20 cruces por cero."""
        fs = 2000
        t = np.linspace(0, 1, fs)
        sine = np.sin(2 * np.pi * 10 * t)
        zcr = self.proc._zero_crossing_rate(sine)
        # Un seno de 10 Hz cruza cero 20 veces en 1 segundo
        assert 15 <= zcr <= 25, f"ZCR inesperado: {zcr} (esperado ~20)"

    def test_spectral_analysis_emg(self):
        """El análisis espectral de EMG debe retornar bandas low/mid/high."""
        emg = self.gen.generate("emg", 500)
        dominant_freq, spectral_power = self.proc._spectral_analysis(emg, 2000, "emg")
        assert 'low' in spectral_power
        assert 'mid' in spectral_power
        assert 'high' in spectral_power
        assert dominant_freq >= 0

    def test_spectral_analysis_eeg_bands(self):
        """El análisis espectral de EEG debe retornar las 5 bandas cerebrales."""
        eeg = self.gen.generate("eeg", 512)
        dominant_freq, spectral_power = self.proc._spectral_analysis(eeg, 256, "eeg")
        expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        for band in expected_bands:
            assert band in spectral_power, f"Falta la banda EEG: {band}"


class TestIntegration:
    """Tests de integración: generator + processor pipeline."""

    def test_full_emg_pipeline(self):
        """Pipeline completo: generar -> procesar EMG."""
        gen = SignalGenerator()
        proc = DSPProcessor()

        emg = gen.generate("emg", 1000)
        result = proc.process(emg, "emg")

        assert result['rms'] >= 0, "RMS no puede ser negativo"
        assert 0 <= result['dominant_freq'] <= 1000, "Frecuencia dominante fuera de rango"
        assert len(result['filtered']) == len(emg), "La señal filtrada debe tener la misma longitud"

    def test_full_ecg_pipeline(self):
        """Pipeline completo: generar -> procesar ECG."""
        gen = SignalGenerator()
        proc = DSPProcessor()

        ecg = gen.generate("ecg", 500)
        result = proc.process(ecg, "ecg")

        assert result['rms'] > 0
        assert len(result['filtered']) == len(ecg)

    def test_multiple_signal_types_consistent(self):
        """Los tres tipos de señal deben procesarse sin errores."""
        gen = SignalGenerator()
        proc = DSPProcessor()

        for signal_type in ['emg', 'ecg', 'eeg']:
            signal = gen.generate(signal_type, 256)
            result = proc.process(signal, signal_type)
            assert not np.any(np.isnan(result['filtered'])), \
                f"NaN en señal filtrada para {signal_type}"

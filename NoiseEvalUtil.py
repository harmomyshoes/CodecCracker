import math
import numpy as np
from numpy.typing import NDArray
from enum import Enum

def convert_time_to_coefficient(
        t: float, sample_rate: int, decay_threshold: float = None
    ) -> float:
        if decay_threshold is None:
            # Attack time and release time in this transform are defined as how long
            # it takes to step 1-decay_threshold of the way to a constant target gain.
            # The default threshold used here is inspired by RT60.
            decay_threshold = convert_decibels_to_amplitude_ratio(-60)
        return 10 ** (math.log10(decay_threshold) / max(sample_rate * t, 1.0))

    
def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)

def get_max_abs_amplitude(samples: NDArray):
    min_amplitude = np.amin(samples)
    max_amplitude = np.amax(samples)
    max_abs_amplitude = max(abs(min_amplitude), abs(max_amplitude))
    return max_abs_amplitude


def calculate_rms(samples):
    """Calculates the root-mean-square value of the audio samples"""
    "return the RMS value in power level"
    return np.sqrt(np.mean(samples**2))

def calculate_rms_dB(samples):
    """Calculates the root-mean-square value of the audio samples"""
    "return the RMS value in power level"
    rms = np.sqrt(np.mean(samples**2))
    return 20 * np.log10(rms * np.sqrt(2))


class MixingType(Enum):
    File = 1
    Track = 2
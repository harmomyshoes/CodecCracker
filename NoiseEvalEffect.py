from audiomentations import Lambda
import numpy as np
from numpy.typing import NDArray
from cylimiter import Limiter as CLimiter
import NoiseEvalUtil as NoiseEvalUtil



##The simple method that create the clipping ratio in fixed
def ClippingDistortionWithFloatingThreshold(samples, sample_rate, clipping_rate):
    clipping_rate = round(clipping_rate, 1)
    lower_percentile_threshold = clipping_rate / 2
    lower_threshold, upper_threshold = np.percentile(
            samples, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
    samples = np.clip(samples, lower_threshold, upper_threshold)
    return samples

###This Method Only using Test Climiter with out the Delay setting
def Dynamic_FullPara_BClimiter(samples,srate,threshold_db,attack_seconds,release_seconds):
    print("Running Limiter")
    attack = NoiseEvalUtil.convert_time_to_coefficient(
        attack_seconds, srate
    )
    release = NoiseEvalUtil.convert_time_to_coefficient(
        release_seconds, srate
    )
    # instead of delaying the signal by 60% of the attack time by default
    delay = 1
    #delay = max(round(0.6 * attack_seconds * srate), 1)
    threshold_factor = NoiseEvalUtil.get_max_abs_amplitude(samples)
    threshold_ratio  = threshold_factor * NoiseEvalUtil.convert_decibels_to_amplitude_ratio(threshold_db)
    
    #print(f"applied configuration attack:{attack},release:{release},threshold:{threshold_ratio},delay:{delay}")
    limiter = CLimiter(
        attack=attack,
        release=release,
        delay=delay,
        threshold=threshold_ratio,
    )
    samples = limiter.limit(samples)
    return samples,srate
import parselmouth as pm
from parselmouth.praat import call
import pandas as pd
import os

from pprint import pprint

# test_sound = pm.Sound("HC_AH\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav")
# pitch = test_sound.to_pitch()
# print(pitch)

# audio_files = pd.read_csv('acoustic-feature-models\audio_files.csv')

def measurePitch(voiceID, f0min, f0max, unit):
    sound = pm.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # std of pitch
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return {
        "mean_pitch": meanF0,
        "std_pitch": stdevF0,
        "hnr": hnr,
        "local_jitter": localJitter,
        "local_abs_jitter": localabsoluteJitter,
        "rap_jitter": rapJitter,
        "ppq5_jitter": ppq5Jitter,
        "ddp_jitter": ddpJitter,
        "local_shimmer": localShimmer,
        "local_db_shimmer": localdbShimmer,
        "apq3_shimmer": apq3Shimmer,
        "aqpq5_shimmer": aqpq5Shimmer,
        "apq11_shimmer": apq11Shimmer,
        "dda_shimmer": ddaShimmer
    }


pprint(measurePitch("HC_AH\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav", 50, 500, "Hertz"))

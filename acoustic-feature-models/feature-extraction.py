import parselmouth as pm
from parselmouth.praat import call
import pandas as pd
import statistics

from pprint import pprint

def getFormants(sound, point_process):
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    num_points = call(point_process, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, num_points):
        point += 1
        t = call(point_process, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)
    
    return {
        "num_points": num_points,
        "f1_mean": f1_mean,
        "f2_mean": f2_mean,
        "f3_mean": f3_mean,
        "f4_mean": f4_mean
    }

def measurePitch(voice_ID, f0min, f0max, unit):
    sound = pm.Sound(voice_ID) # read the sound
    # Pitch-related things
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    mean_F0 = call(pitch, "Get mean", 0, 0, unit) # mean pitch
    std_F0 = call(pitch, "Get standard deviation", 0 ,0, unit) # std of pitch
    min_F0 = call(pitch, "Get minimum", 0, 0, unit, "Parabolic") # min pitch
    max_F0 = call(pitch, "Get maximum", 0, 0, unit, "Parabolic") # max pitch
    pitch_range = max_F0 - min_F0 # pitch range
    # Hnr
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0) # mean hnr
    # Jitter
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_abs_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    # Shimmer
    local_shimmer =  call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    local_db_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer =  call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # Intensity
    intensity = call(sound, "To Intensity", f0min, 0, "yes")
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    std_intensity = call(intensity, "Get standard deviation", 0, 0)
    min_int = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_int = call(intensity, "Get maximum", 0, 0, "Parabolic")
    intensity_range = max_int - min_int

    pprint(getFormants(sound, point_process))
    
    return {
        "mean_pitch": mean_F0,
        "std_pitch": std_F0,
        "range_pitch": pitch_range,
        "mean_hnr": hnr,
        "local_jitter": local_jitter,
        "local_abs_jitter": local_abs_jitter,
        "rap_jitter": rap_jitter,
        "ppq5_jitter": ppq5_jitter,
        "ddp_jitter": ddp_jitter,
        "local_shimmer": local_shimmer,
        "local_db_shimmer": local_db_shimmer,
        "apq3_shimmer": apq3_shimmer,
        "aqpq5_shimmer": aqpq5_shimmer,
        "apq11_shimmer": apq11_shimmer,
        "dda_shimmer": dda_shimmer,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "range_intensity": intensity_range,
    }


pprint(measurePitch("HC_AH\AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav", 50, 500, "Hertz"))

import scipy.io as io
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
import numpy as np
import scipy.linalg as lin

def saveSignalAsWAV(name, signal, fs):
    
    io.wavfile.write(name, fs, signal)

def getOriginalSignals():

    path = ["musicSignal.wav", "voiceSignal.wav", "chirpLinearSignal.wav",
            "chirpExpSignal.wav", "noiseSignal.wav", "squareSignal.wav"]

    fs1, musicSignal = io.wavfile.read(path[0])
    fs2, voiceSignal = io.wavfile.read(path[1])
    fs3, chirpLinearSignal = io.wavfile.read(path[2])
    fs4, chirpExpSignal = io.wavfile.read(path[3])
    fs5, gaussSignal = io.wavfile.read(path[4])
    fs6, squareSignal = io.wavfile.read(path[5])

    fs = {"music": fs1, "voice": fs2, "chirpLin": fs3,
        "chirpExp": fs4, "gauss": fs5, "square": fs6}
    signal = {"music": musicSignal, "voice": voiceSignal, "chirpLin": chirpLinearSignal,
                "chirpExp": chirpExpSignal, "gauss": gaussSignal, "square": squareSignal}

    return fs, signal, path

def getRecordedSignals():

    path = ["recmusicSignal.wav", "recVoiceSignal.wav", "recchirpLinearSignal.wav",
            "recchirpExpSignal.wav", "recnoiseSignal.wav", "recsquareSignal.wav"]

    fs1, recmusicSignal = io.wavfile.read(path[0])
    fs2, recVoiceSignal = io.wavfile.read(path[1])
    fs3, recchirpLinearSignal = io.wavfile.read(path[2])
    fs4, recchirpExpSignal = io.wavfile.read(path[3])
    fs5, recgaussSignal = io.wavfile.read(path[4])
    fs6, recsquareSignal = io.wavfile.read(path[5])

    recmusicSignal = (recmusicSignal.T)[1]
    recVoiceSignal = (recVoiceSignal.T)[1]
    recchirpLinearSignal = (recchirpLinearSignal.T)[1]
    recchirpExpSignal = (recchirpExpSignal.T)[1]
    recgaussSignal = (recgaussSignal.T)[1]
    recsquareSignal = (recsquareSignal.T)[1]

    fs = {"music": fs1, "voice": fs2, "chirpLin": fs3,
        "chirpExp": fs4, "gauss": fs5, "square": fs6}
    signal = {"music": recmusicSignal, "voice": recVoiceSignal, "chirpLin": recchirpLinearSignal,
          "chirpExp": recchirpExpSignal, "gauss": recgaussSignal, "square": recsquareSignal}

    return fs, signal, path

def play(signal, fs):
    audio = ipd.Audio(signal, rate=fs, autoplay=True)
    return audio

def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = sp.spectrogram(w, fs=fs, nperseg=256, nfft=576)
    fig, ax = plt.subplots()
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r',
                  shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)

def getNextPowerOfTwo(len):
    return 2**(len*2).bit_length()

def get_optimal_params(x, y, M):
    
    N = len(x)
    r = sp.correlate(x, x)/N
    p = sp.correlate(x, y)/N
    r = r[N-1:N-1 + M]
    p = p[N-1:N-1-(M):-1]           # Correlate calcula la cross-corr r(-k), y necesitamos r(k), y esto no es par como la autocorrelacion
    wo = lin.solve_toeplitz(r, p)

    jo = np.var(y) - np.dot(p, wo)

    NMSE = jo/np.var(y)
    
    return wo, jo, NMSE



import numpy as np

def convert_to_kspace(image):
    return np.fft.fftshift(np.fft.fft2(image).real)
import numpy as np

def convert_to_kspace(image):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image), axes=(0, 1, 2)).real)
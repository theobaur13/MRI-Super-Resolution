import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

def convert_to_kspace(image):
    kspace = fftshift(fftn(ifftshift(image), axes=(0, 1, 2)))
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, [0, 1, 2])))
    return kspace

def convert_to_image(kspace):
    image = fftshift(ifftn(ifftshift(kspace), axes=(0, 1, 2)))
    image *= np.sqrt(np.prod(np.take(image.shape, [0, 1, 2])))
    return image

def undersampling(kspace, axis, factor, method="random"):
    if method == "random":
        return random_undersampling(kspace, factor)
    elif method == "linear":
        return linear_undersampling(kspace, axis, factor)
    elif method == "radial":
        return radial_undersampling(kspace, axis)

def random_undersampling(kspace, factor):
    mask = np.random.choice([0, 1], size=kspace.shape, p=[1 - 1 / factor, 1 / factor])
    return kspace * mask

def linear_undersampling(kspace, axis, factor):
    # Create a mask that keeps every x-th line along the specified axis
    mask = np.zeros(kspace.shape)
    slices = [slice(None)] * 3
    slices[axis] = slice(None, None, factor)  # Slice every x-th line
    
    mask[tuple(slices)] = 1
    return kspace * mask

def radial_undersampling(kspace, axis, radius=50):
    # Create a mask that keeps only the center of the k-space, where the axis is the axis of slicing
    mask = np.zeros(kspace.shape)
    center = np.array(kspace.shape) // 2

    Z, Y, X = np.indices(kspace.shape)

    if axis == 0:       # Cylinder along the z-axis
        mask = np.where(np.sqrt((X - center[2])**2 + (Y - center[1])**2) <= radius, 1, 0)

    elif axis == 1:     # Cylinder along the y-axis
        mask = np.where(np.sqrt((X - center[2])**2 + (Z - center[0])**2) <= radius, 1, 0)

    elif axis == 2:     # Cylinder along the x-axis
        mask = np.where(np.sqrt((Y - center[1])**2 + (Z - center[0])**2) <= radius, 1, 0)

            
    return kspace * mask
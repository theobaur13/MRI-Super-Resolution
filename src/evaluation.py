import numpy as jnp

def psnr(original, compressed):
    mse = jnp.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # No noise, perfect match
    max_pixel = 255.0
    psnr_value = 20 * jnp.log10(max_pixel / jnp.sqrt(mse))
    return round(psnr_value.real, 3)
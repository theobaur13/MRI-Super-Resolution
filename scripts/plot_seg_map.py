import matplotlib.pyplot as plt
import matplotlib.colors 
import nibabel as nib
import numpy as np

scan = "BraTS-GLI-01762-000-t2f_sr.nii.gz"
bg_nifti_path = f"E:\\ESRGAN_RRDB3_triple\\seg_eval\\input\\{scan}"
seg_nifti_path = f"E:\\ESRGAN_RRDB3_triple\\seg_eval\\output_tumor\\{scan}"
slice_index = 65

bg_img = nib.load(bg_nifti_path).get_fdata()[..., slice_index]
seg_img = nib.load(seg_nifti_path).get_fdata()[..., slice_index]

plt.figure(figsize=(8, 8))
plt.imshow(bg_img, cmap='gray')
masked_seg = np.ma.masked_where(seg_img == 0, seg_img)
cmap = matplotlib.colors.ListedColormap(['red'])
plt.imshow(masked_seg, alpha=0.5, cmap=cmap)
plt.axis('off')
plt.show()
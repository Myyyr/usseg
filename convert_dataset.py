import nibabel as nib
import numpy as np
import os

import sys


L=  ['17L', '53L', '42R', '84L', '47R', '33L', '14L', '94R', '119R', '43L', '54R', '23L', '11R', '116R', '12L', '116L', '38R', '38L', '51R']

def main(argv, arc):

	path = argv[1]
	out_path = argv[2]
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	for f in os.listdir(path):
		b = False
		for l  in L:
			b = (l in f)
		if (".nii" in f) and not b:
			x = nib.load(os.path.join(path, f)).get_fdata()
			out_f = f.replace(".nii.gz", ".npz")
			np.savez(os.path.join(out_path, out_f), x)





if __name__ == '__main__':
	main(sys.argv, len(sys.argv))


# /scratch/lthemyr/20220318_US_DATA/USimg_cropped
# /scratch/lthemyr/20220318_US_DATA/USimg_cropped_npz

# /scratch/lthemyr/20220318_US_DATA/USmask_cropped
# /scratch/lthemyr/20220318_US_DATA/USmask_cropped_npz

# /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128
# /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128_npz

# /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128
# # /scratch/lthemyr/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128_npz


# # ------------

# # /etudiants/siscol/t/themyr_l/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128
# # /etudiants/siscol/t/themyr_l/US/us_3d_segmentation_dataset_08_03_2022/USimg_cropped128_npz

# /etudiants/siscol/t/themyr_l/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128
# /etudiants/siscol/t/themyr_l/US/us_3d_segmentation_dataset_08_03_2022/USmask_cropped128_npz
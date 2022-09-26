import numpy as np
from monai.metrics import compute_meandice
import nibabel as nib
import torch
import monai.transforms as T

import argparse
import os
import json



def main(pred_pth, gt_pth, out_pth):
	if out_pth == "":
		out_pth = pred_pth
	out_file = "final_results.json"

	for fp in os.listdir(pred_pth):
		if ".npz" in fp:
			fg = fp.replace('pred.npz', 'Vol.nii.gz')

			pred = np.load(os.path.join(pred_pth, fp))['arr_0']
			gt   = nib.load(os.path.join(gt_pth, fg)).get_fdata()
			size = gt.shape
			print("a", gt.shape, pred.shape)

			pred = torch.from_numpy(pred)
			pred = T.Resize(size, mode="nearest")(pred[None, ...])[0,...]
			print("b", gt.shape, pred.shape)

			pred = pred[0,...].numpy()

			print("c", gt.shape, pred.shape)


			# Compute dice and hausdorff in a json !!!







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pred_pth', help='path of the predictions')
	parser.add_argument('gt_pth', help='path of the ground truth', default="/scratch/lthemyr/20220318_US_DATA/USmask_cropped", required=False)
	parser.add_argument('out_pth', help='path of the output file', default="", required=False)

	args = parser.parse_args()

	main(args.pred_pth, args.gt_pth, args.out_pth)
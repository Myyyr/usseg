import numpy as np
from monai.metrics import compute_meandice, compute_hausdorff_distance
import nibabel as nib
import torch
import monai.transforms as T
from scipy.ndimage import zoom
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched

import argparse
import os
import json



def main(pred_pth, gt_pth, out_pth):
	classes = 2
	if out_pth == "":
		out_pth = pred_pth
	out_file = "final_results.json"

	for fp in os.listdir(pred_pth):
		if ".npz" in fp:
			fg = fp.replace('pred.npz', 'Vol.nii.gz')

			pred = np.load(os.path.join(pred_pth, fp))['arr_0'][0,...]
			gt   = nib.load(os.path.join(gt_pth, fg)).get_fdata()
			print("a.1", gt.shape, pred.shape)

			gt = zoom(gt, (0.3, 0.3, 0.3))
			size = gt.shape
			print("a.2", gt.shape, pred.shape)

			pred = torch.from_numpy(pred)
			pred = T.Resize(size, mode="nearest")(pred[None, ...])#[0,...]
			print("b", gt.shape, pred.shape)

			pred = pred[0,...].numpy()

			print("c", gt.shape, pred.shape)

			pred = convert_seg_image_to_one_hot_encoding_batched(pred[None, ...], [i for i in range(classes)])
			gt   = convert_seg_image_to_one_hot_encoding_batched(gt[None, ...],   [i for i in range(classes)])

			print("d", gt.shape, pred.shape)


			pred = torch.from_numpy(pred).float().cuda(0)
			gt   = torch.from_numpy(gt).float().cuda(0)

			dsc = compute_meandice(pred, gt, ignore_empty=False)
			hd95 = compute_hausdorff_distance(pred, gt, percentile=95)

			dsc = dsc.cpu().numpy()[0]
			hd95 = hd95.cpu().numpy()[0]

			print("\n\ndsc", dsc, hd95)
			exit(0)


			# Compute dice and hausdorff in a json !!!







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('pred_pth', help='path of the predictions')
	parser.add_argument('gt_pth', help='path of the ground truth', default="/scratch/lthemyr/20220318_US_DATA/USmask_cropped")
	parser.add_argument('out_pth', help='path of the output file', default="")

	args = parser.parse_args()

	main(args.pred_pth, args.gt_pth, args.out_pth)
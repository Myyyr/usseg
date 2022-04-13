from monai.data import Dataset
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous, LoadImage
import random
from CustomTransform import CustomRandCropByPosNegLabeld
from monai.transforms import RandCropByPosNegLabeld, Compose
# from tools import log_debug, log_info, log_start, log_end
import numpy as np

from tools import downsample_seg_for_ds_transform3
from einops import rearrange

import time

class CustomDataset(Dataset):
	def __init__(self, data, transform=None, iterations=250, crop_size=[128,128,128], log=None, net_num_pool_op_kernel_sizes=None, val=False, *args, **kwargs):
		# We use our own Custom dataset wich with we can keep track of sub volumes position.
		self.data = Dataset(data)
		self.iterations = iterations
		self.loader = LoadImage()
		self.n_data = len(data)
		self.transform = transform
		self.log=log
		self.val=val
		self.croper = CustomRandCropByPosNegLabeld(
						            keys=["image", "label"],
						            label_key="label",
						            spatial_size=crop_size,
						            pos=1,
						            neg=1,
						            num_samples=1,
						            image_key="image",
						            image_threshold=0,
						            log=log
						        )
		self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
		self.idx = -1

	def __len__(self):
		if not self.val:
			return self.iterations
		else:
			return len(self.data)

	def __getitem__(self, index):
		t0 = time.time()
		log=self.log
		# To Do: dataset when eval mode
		if not self.val:
			i = random.randint(0,self.n_data-1)
		else:
			self.idx += 1
			i = self.idx

		# log.debug("data_i['label'].shape", data_i["label"].shape)
		# log.debug("index", index)
		# log.debug("i", i)
		data_i = {}
		data_i["image"] = rearrange(np.load(self.data[i]["image"])['arr_0'][None, ...], 'b x y z -> b z x y')
		t1 = time.time()
		data_i["label"] = rearrange(np.load(self.data[i]["label"])['arr_0'][None, ...], 'b x y z -> b z x y')
		t2 = time.time()
		data_i["id"] = [self.data[i]["image"].split('/')[-1].replace('img', 'xxx')]

		shape = data_i["image"].shape
		if not self.val:
			data_i, centers = self.croper(data_i)
			t3 = time.time()
			data_i = data_i[0]
			# log.debug("index", index)
			# log.debug("centers", centers[0])
			centers = [centers[0][2]-shape[3]//2,centers[0][0]-shape[1]//2,centers[0][1]-shape[2]//2]
			# log.debug("centers", centers)
			# log.debug("shape", shape)
			data_i["center"] = np.array(centers)


			# Apply transformations
			data_i = apply_transform(self.transform, data_i) if self.transform is not None else data_i
			t4 = time.time()



		# Do deep supervision on labels
		if self.net_num_pool_op_kernel_sizes!=None:
			deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
	            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

			data_i["label"] = downsample_seg_for_ds_transform3(data_i["label"][None,...], deep_supervision_scales, classes=[0,1])
		# log.debug("loadok")
		t5 = time.time()

		tim=t1-t0
		tla=t2-t1
		tcr=t3-t2
		ttr=t4-t3
		tde=t5-t4
		tal=t5-t0


		log.debug("Dataset times", "im:{}s | lab:{}s | crop:{}s | trans:{}s | deep:{}s | ALL:{}s |".format(tim, tla, tcr, ttr, tde, tal))
		return data_i


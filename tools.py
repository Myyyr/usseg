import os
import importlib
from termcolor import colored

import numpy as np
import torch
from torch.nn.functional import avg_pool3d

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched


def create_path_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def downsample_seg_for_ds_transform3(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), classes=None):
    output = []
    one_hot = torch.from_numpy(convert_seg_image_to_one_hot_encoding_batched(seg[:, 0], classes)) # b, c,

    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(torch.from_numpy(seg[0,...]))
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i-1) // 2 for i in kernel_size)
            pooled = avg_pool3d(one_hot, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)

            output.append(pooled[0,...])
    return output

def get_loss(net_num_pool_op_kernel_sizes):
	################# Here we wrap the loss for deep supervision ############
	# we need to know the number of outputs of the network
	loss = DC_and_CE_loss({'batch_dice':True, 'smooth': 1e-5, 'do_bg': False}, {}) #maybe batch dice false
	net_numpool = len(net_num_pool_op_kernel_sizes)

	# we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
	# this gives higher resolution outputs more weight in the loss
	weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

	# we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
	mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
	weights[~mask] = 0
	weights = weights / weights.sum()
	ds_loss_weights = weights
	# now wrap the loss
	loss = MultipleOutputLoss2(loss, ds_loss_weights)
	################# END ###################

	return loss



def import_model(name, *args, **kwargs):
	return importlib.import_module("models."+name).model(**kwargs)



def create_split(im_pth, seg_pth, split):
	splits = []
	files = sorted(os.listdir(im_pth))

	for spl in split:
		tmp = {
				'image': os.path.join(im_pth,files[spl]),
				'label': os.path.join(seg_pth,files[spl].replace("img", "Vol")),
				'id': spl
			}
		splits.append(tmp)

	return splits


class Log(object):
	"""docstring for Log"""
	def __init__(self, log):
		super(Log, self).__init__()
		self.log = log
		

	def debug(self, msg, info=""):
		self.log.debug(colored(msg+"\n", "red")+colored(str(info), "magenta"))

	def info(self, msg, info):
		self.log.info(colored(msg+"\n", "blue")+colored(str(info), "green"))
		
	def start(self, msg):
		self.log.info(colored("Start "+msg+" ...\n", "blue"))

	def end(self, msg):
		self.log.info(colored("... "+msg+" Done\n", "blue"))
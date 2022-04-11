import os
import torch
import monai
from monai.data import Dataset
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.transforms import (
	Compose,
	RandRotated,
	Orientationd,
	SaveImaged,
	ScaleIntensityRanged,
	Spacingd,
	EnsureTyped,
	EnsureType,
	Invertd,
	RandAdjustContrastd,
	RandFlipd,
	LoadImage
)
# from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice, compute_hausdorff_distance

from tools import create_split, import_model, get_loss, poly_lr, create_path_if_not_exists
from CustomTransform import CustomRandScaleCropd
from CustomDataset import CustomDataset

from nnunet.utilities.nd_softmax import softmax_helper

from tqdm import tqdm

import nibabel as nib
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import telegram_send as ts

class Trainer():
	def __init__(self, cfg, log, *args, **kwargs):
		# Logs
		self.log = log
		self.dbg=cfg.training.dbg
		self.writer = SummaryWriter(log_dir='tensorboard/'+cfg.dataset.name+'_'+cfg.training.name+'_'+cfg.model.name)
		self.dataset_name = cfg.dataset.name
		self.training_name = cfg.training.name
		self.model_name = cfg.model.name
		

		# Device
		os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.training.gpu)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.use_gpu = cfg.training.use_gpu

		# Fix seed : TO DO

		# Hyperparameters
		log.debug("Hyperparameters")
		self.epochs = cfg.training.epochs
		self.start_epoch = 0
		self.initial_lr = cfg.training.lr
		self.batch_size = cfg.training.batch_size
		self.num_workers = cfg.training.num_workers
		self.crop_size = cfg.training.crop_size
		self.iterations = cfg.training.iter
		self.weight_decay = cfg.training.weight_decay
		self.net_num_pool_op_kernel_sizes = cfg.training.net_num_pool_op_kernel_sizes
		self.net_conv_kernel_sizes = cfg.training.net_conv_kernel_sizes

		# Dataset
		log.debug("Dataset")
		self.seg_path = cfg.dataset.path.seg
		self.train_split = create_split(cfg.dataset.path.im, cfg.dataset.path.seg, cfg.dataset.split.train)
		self.val_split   = create_split(cfg.dataset.path.im, cfg.dataset.path.seg, cfg.dataset.split.val)

		train_transforms = None
		val_transforms = None

		train_transforms = Compose([
					RandRotated(keys=["image", "label"], 
								range_x=cfg.training.augmentations.rotate.x_, 
								range_y=cfg.training.augmentations.rotate.y_, 
								range_z=cfg.training.augmentations.rotate.z_, 
								prob=cfg.training.augmentations.rotate.p_),
					# CustomRandScaleCropd(keys=["image", "label"],
					# 					 roi_scale=cfg.training.augmentations.scale.min_,
					# 					 max_roi_scale=cfg.training.augmentations.scale.max_,
					# 					 prob=1,)#cfg.training.augmentations.scale.p_,)
					# 					 # random_size=False),
					RandAdjustContrastd(keys=["image", "label"],
										prob=cfg.training.augmentations.gamma.p_,
										gamma=cfg.training.augmentations.gamma.g_),
					
			])

		trainData = CustomDataset(self.train_split, transform=train_transforms, iterations=self.iterations, crop_size=self.crop_size, log=log, net_num_pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes) # Add transforms : TO DO
		valData   = CustomDataset(self.val_split,   transform=val_transforms, iterations=self.iterations, crop_size=self.crop_size, log=log, val=True) # Add transforms : TO DO

		self.train_loader = DataLoader(trainData, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
		self.val_loader = DataLoader(valData, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

		self.stride = cfg.training.inference.stride
		self.classes = cfg.dataset.classes

		# Models
		log.debug("Model")
		self.save_path = create_path_if_not_exists(cfg.model.checkpoint.path)
		self.n_save = cfg.model.checkpoint.save
		self.do_load_checkpoint = cfg.model.checkpoint.load
		self.load_path = cfg.model.checkpoint.load_path

		self.model = import_model(cfg.model.model, dataset='US', num_classes=self.classes, 
													num_pool=len(self.net_num_pool_op_kernel_sizes), 
													pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
													conv_kernel_sizes=self.net_conv_kernel_sizes)

		if torch.cuda.is_available() and self.use_gpu:
			self.model.cuda()
		self.model.inference_apply_nonlin = softmax_helper

		self.lr = self.initial_lr

		self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, weight_decay=self.weight_decay,
										 momentum=0.99, nesterov=True)

		
		self.loss = get_loss(self.net_num_pool_op_kernel_sizes)

		self.infer_path = cfg.model.inference.path
		if not os.path.exists(self.infer_path):
			os.makedirs(self.infer_path)


		if self.do_load_checkpoint:
			self.load_checkpoint()



		



	def run_training(self, *args, **kwargs):
		log=self.log
		if not self.dbg:
			ts.send(messages=["Training: " + self.dataset_name+'_'+self.training_name+'_'+self.model_name])


		for epoch in range(self.start_epoch, self.epochs):
			self.model.train()
			self.optimizer.param_groups[0]['lr'] = self.lr
			# log.debug("Memory", torch.cuda.max_memory_allocated()//1024**3)
			btc = -1
			for batch_data in tqdm(self.train_loader):
				btc+=1
				self.optimizer.zero_grad()

				inputs = batch_data["image"]
				labels = batch_data["label"]
				centers = batch_data["center"]
				
				# log.debug("inputs.shape", inputs.shape)
				# log.debug("type(inputs)", type(inputs))
				# log.debug("labels[0].shape", labels[0].shape)
				# log.debug("centers", centers)

				

				if torch.cuda.is_available() and self.use_gpu:
					inputs = inputs.cuda(0)
					# log.debug("Ep:{}:Btc:{} Input".format(epoch, btc), "Mem: {} Gb | Shape: {}".format(torch.cuda.max_memory_allocated()//1024**3, inputs.shape) )
					
					# labels = [lab.cuda(0) for lab in labels]
					for lab in range(len(labels)):
						# log.debug("Ep:{}:Btc:{} Labels".format(epoch, btc), "Shape: {}".format(labels[lab].shape))
						labels[lab] = labels[lab].cuda(0)
						# log.debug("Ep:{}:Btc:{} Labels".format(epoch, btc), "Mem: {} Gb".format(torch.cuda.max_memory_allocated()//1024**3))


					# centers.cuda()
					# log.debug("torch.cuda.is_available() and self.use_gpu")
				# log.debug("inputs.device",inputs.device)
				# log.debug("model.device",self.model.device)

				output = self.model(inputs, centers)
				del inputs
				# log.debug("type(output)", type(output))
				l = self.loss(output, labels)
				l.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
				self.optimizer.step()

				for out in output:
					del out
				del output
				for lab in labels:
					del lab
				del labels

			saved_txt = ""
			if (epoch+1)%self.n_save == 0:
				self.save_chekpoint(epoch)
				save_chekpoint = " :: Saved!"
			log.info("Epoch: {}".format(epoch), "Loss: {}, lr: {}{}".format(l.detach().cpu().numpy(),
																		self.lr,
																		save_chekpoint
																		))
			self.writer.add_scalar('Loss', l.detach().cpu().numpy(), epoch)
			self.writer.add_scalar('lr', self.lr, epoch)
			self.lr = poly_lr(epoch, self.epochs, self.initial_lr, 0.9)
			torch.cuda.empty_cache()

		if not self.dbg:
			ts.send(messages=["Training END: " + self.dataset_name+'_'+self.training_name+'_'+self.model_name])


	def run_eval(self, do_infer=True, *args, **kwargs):
		log=self.log

		if do_infer:
			self.model.eval()
			for batch_data in tqdm(self.val_loader):
				inputs = batch_data["image"]
				prediction = self.inference(inputs)
				prediction = torch.argmax(prediction, dim=1)
				idx = batch_data["id"][0][0]

				name = idx.replace('xxx', 'pred')
				file = os.path.join(self.infer_path, name)

				pred_nib = nib.Nifti1Image(prediction.numpy(), None)
				nib.save(pred_nib, file)

		loader = LoadImage()
		results = {}
		mean_all = None
		N = 0

		for f in os.listdir(self.infer_path):
			if '.nii' in f:
				pred = loader(os.path.join(self.infer_path, f))[0][0]
				anno = loader(os.path.join(self.seg_path, f.replace('pred', 'Vol')))[0]

				pred = convert_seg_image_to_one_hot_encoding_batched(pred[None, ...], [i for i in range(self.classes)])
				anno = convert_seg_image_to_one_hot_encoding_batched(anno[None, ...], [i for i in range(self.classes)])

				pred = torch.from_numpy(pred)
				anno = torch.from_numpy(anno)

				dice = compute_meandice(anno, pred)
				hd95 = compute_hausdorff_distance(anno, pred, percentile=95)

				dice = dice.numpy()[0]
				hd95 = hd95.numpy()[0]

				idx = f.replace('pred', 'xxx')
				log.info(idx, "Dice: {}\nHD95: {}".format(dice, hd95))

				results[idx] = {'dice':dice.tolist(), 'hd95':hd95.tolist()}

				if mean_all == None:
					mean_all = {'dice':np.array(dice), 'hd95':np.array(hd95)}
				else:
					mean_all['dice'] += np.array(dice)
					mean_all['hd95'] += np.array(hd95)
				N+=1

		mean_all['dice'] = (mean_all['dice']/N).tolist()
		mean_all['hd95'] = (mean_all['hd95']/N).tolist()
		results['mean_all'] = mean_all

		log.info("mean all", "Dice: {}\nHD95: {}".format(mean_all['dice'], mean_all['hd95']))

		with open(os.path.join(self.infer_path,'results.json'), 'w') as outfile:
			json_string = json.dumps(results)
			outfile.write(json_string)



	def save_chekpoint(self, epoch):
		state_dict = self.model.state_dict()
		optimizer_state_dict = self.optimizer.state_dict()
		save_this = {
			'epoch': epoch + 1,
			'state_dict': state_dict,
			'optimizer_state_dict': optimizer_state_dict}
		torch.save(save_this, os.path.join(self.save_path, "latest.pt"))

	def load_checkpoint(self):
		
		checkpoint = torch.load(self.load_path)
		self.model.load_state_dict = checkpoint['state_dict']
		self.start_epoch = checkpoint['epoch']
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


	def inference(self, inputs):
		log=self.log

		B, C, H, W, D = inputs.shape
		H_crop, W_crop, D_crop = self.crop_size
		# log.debug("inputs.shape", inputs.shape)

		nH, nW, nD = int(H//(H_crop*self.stride[0])), int(W//(W_crop*self.stride[1])), int(D//(D_crop*self.stride[2]))

		output = torch.zeros((B, self.classes, H, W, D))
		count  = torch.zeros((B, self.classes, H, W, D))


		for i in range(nH):
			for j in range(nW):
				for k in range(nD):
					idx_h = int(i*H_crop*self.stride[0])
					idx_w = int(j*W_crop*self.stride[1])
					idx_d = int(k*D_crop*self.stride[2])

					if idx_h+H_crop > H:
						idx_h = H - H_crop
					if idx_w+W_crop > W:
						idx_w = W - W_crop
					if idx_d+D_crop > D:
						idx_d = D - D_crop

					crop = inputs[:,:,idx_h:idx_h+H_crop, idx_w:idx_w+W_crop, idx_d:idx_d+D_crop]
					centers = [[idx_h+H_crop//2, idx_w+W_crop//2, idx_d+D_crop//2] for i in range(B)]
					out_crop = self.model(crop, centers)

					del crop

					output[:,:,idx_h:idx_h+H_crop, idx_w:idx_w+W_crop, idx_d:idx_d+D_crop] = out_crop[0].cpu()
					count[:,:,idx_h:idx_h+H_crop, idx_w:idx_w+W_crop, idx_d:idx_d+D_crop]  += 1


		return output/count









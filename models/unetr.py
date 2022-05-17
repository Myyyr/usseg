from monai.networks.nets.unetr import UNETR 
from nnunet.network_architecture.neural_network import SegmentationNetwork


class model(SegmentationNetwork):
	def __init__(self, in_channels=1, num_classes=1, img_size=[128,128,128],*args, **kwargs):
		super(model, self).__init__()
		self.network = UNETR(in_channels=in_channels, out_channels=num_classes, img_size=img_size)

	def forward(self, x, *args, **kwargs):
		return self.network(x)








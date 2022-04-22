from monai.networks.nets import vnet 
from nnunet.network_architecture.neural_network import SegmentationNetwork


class model(SegmentationNetwork):
	def __init__(self,spatial_dims=3, in_channels=1, num_classes=2, *args, **kwargs):
		self.network = vnet.VNet(out_channels=num_classes)

	def forward(self, x, *args, **kwargs):
		return self.network(x)








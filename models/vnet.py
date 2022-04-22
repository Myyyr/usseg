from monai.networks.nets import vnet 


class model(SegmentationNetwork):
	def __init__(self,spatial_dims=3, in_channels=1, num_classes=2, *args, **kwargs):
		self.network = vnet(out_chanels=num_classes)

	def forward(self, x, *args, **kwargs):
		return self.network(x)








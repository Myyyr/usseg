import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import torch

log = logging.getLogger(__name__)




@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:

	log.info("Config:\n{}".format(OmegaConf.to_yaml(cfg)))
	log.info("Working directory:\n{}".format(os.getcwd()))
	log.debug("Debug level message")

	log.debug("GPUs:\n{}".format(torch.cuda.device_count()))
	



if __name__ == '__main__':
	main()
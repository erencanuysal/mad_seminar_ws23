from models.ae import AE
from models.vae import VAE
#from models.ra import RA


def get_model(config):
    print(f"Loading model {config['model_name']}")
    if config['model_name'] == 'AE':
        return AE(config)
    elif config['model_name'] == 'VAE':
        return VAE(config)
 #   elif config['model_name'] == 'RA':
  #      return RA(config)
    else:
        raise ValueError(f"Unknown model name {config['model_name']}")

import torch
import torch.nn as nn
from models.LGFMUNet.UNet_VMamba import LGMUNet

class LGMUNet_Factory(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 encoder_depths=[3, 3, 3, 3],
                 decoder_depths=[3, 3, 3],
                 embed_dim=96,
                 d_state=16,
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 deep_supervision=False):
        """
        LGMUNet Factory

        Args:
            input_channels (int): Number of input image channels
            num_classes (int): Number of segmentation classes
            encoder_depths (list): Encoder stage depths
            decoder_depths (list): Decoder stage depths
            embed_dim (int): Embedding dimension
            d_state (int): Mamba state dimension
            drop_path_rate (float): Drop path probability
            load_ckpt_path (str): Pretrained weights path
            deep_supervision (bool): Whether to use deep supervision
        """
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Create config object (replacing the original dict)
        # Create a simple object to simulate the original config structure
        class ConfigObject:
            pass

        config = ConfigObject()
        config.hyper_parameter = ConfigObject()
        config.hyper_parameter.crop_size = (448, 448)
        config.hyper_parameter.convolution_stem_down = 8
        config.hyper_parameter.blocks_num = encoder_depths
        config.hyper_parameter.drop_rate = drop_path_rate

        # Create LGMUNet model
        self.lgmunet = LGMUNet(
            num_input_channels=input_channels,
            num_classes=num_classes,
            embedding_dim=embed_dim,
            d_state=d_state,
            deep_supervision=deep_supervision,
            config=config
        )

        # Load pretrained weights if available
        if load_ckpt_path:
            self.load_from(load_ckpt_path)

    def forward(self, x):
        # If input is single-channel but model needs 3 channels, duplicate channels
        if x.size()[1] == 1 and self.lgmunet.num_input_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        # Forward pass
        output = self.lgmunet(x)

        '''
        # Apply sigmoid for binary classification tasks
        if self.num_classes == 1 and not self.deep_supervision:
            return torch.sigmoid(output)
        '''
        return output

    def load_from(self, ckpt_path):
 """"""
        if not ckpt_path:
            print("No checkpoint path provided. Skipping weight loading.")
            return

        try:
            # Load the entire model state dict
            model_dict = self.lgmunet.state_dict()
            checkpoint = torch.load(ckpt_path)

            # Handle different checkpoint formats
            if 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint

            # Phase 1: Load encoder weights
            # Find all encoder-related keys
            encoder_keys = [k for k in pretrained_dict.keys() if 'encoder' in k]
            encoder_dict = {k: pretrained_dict[k] for k in encoder_keys}

            # Update encoder weights
            model_dict.update(encoder_dict)
            self.lgmunet.load_state_dict(model_dict, strict=False)
            print(f"Encoder loaded: {len(encoder_dict)} weights transferred")

            # Phase 2: Load decoder weights
            # Create decoder weight mapping (encoder layer -> decoder layer)
            decoder_mapping = {}
            num_encoder_layers = len(self.lgmunet.encoder.layers)

            for i in range(num_encoder_layers - 1):
                src_prefix = f"encoder.layers.{i}"
                tgt_prefix = f"decoder.layers.{num_encoder_layers - 2 - i}"  # Reverse mapping

                # Add all matching weights
                for k, v in pretrained_dict.items():
                    if src_prefix in k:
                        new_key = k.replace(src_prefix, tgt_prefix)
                        decoder_mapping[new_key] = v

            # Update decoder weights
            model_dict.update(decoder_mapping)
            self.lgmunet.load_state_dict(model_dict, strict=False)
            print(f"Decoder loaded: {len(decoder_mapping)} weights transferred")

            # Print unloaded keys
            all_loaded_keys = set(encoder_keys) | set(decoder_mapping.keys())
            not_loaded = set(pretrained_dict.keys()) - all_loaded_keys
            if not_loaded:
                print(f"Not loaded keys: {list(not_loaded)[:5]}... (total {len(not_loaded)})")

        except Exception as e:
            print(f"Error loading weights: {e}")
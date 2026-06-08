from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import random_split
#from configs.config_lvmunet import DataConfig
import torchvision.transforms.functional as TF
from configs.Config_comparative import DataConfig

def denormalize(image, mean, std):
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # (C, H, W) 
    if image.dim() == 3:
        if image.size(0) in [1, 3]:  # (C, H, W) 
            pass
        elif image.size(1) in [1, 3]:  # (H, C, W) 
            image = image.permute(1, 0, 2)
        elif image.size(2) in [1, 3]:  # (H, W, C) 
            image = image.permute(2, 0, 1)

    denorm_img = image.clone().detach()
    for c in range(3):
        denorm_img[c] = denorm_img[c] * std[c] + mean[c]

    # 0-255
    denorm_img = denorm_img * 255

    # (H, W, C) 
    denorm_img = denorm_img.permute(1, 2, 0).byte().numpy()
    return denorm_img


class XJQ_datasets(Dataset):
    def __init__(self, path_Data, config: DataConfig, split='train'):
        super(XJQ_datasets, self).__init__()
        self.config = config
        self.split = split


        images_dir = os.path.join(path_Data, 'train', 'images')
        masks_dir = os.path.join(path_Data, 'train', 'masks')

        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))


 assert len(images_list) == len(masks_list), ""
        self.data = []
        for img_name, mask_name in zip(images_list, masks_list):
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, mask_name)
            self.data.append([img_path, mask_path])


        if split == 'train':
            self.transformer = config.train_transforms
        else:
            self.transformer = config.test_transforms

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        # PIL
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # PIL
        mask = mask.point(lambda p: 255 if p > 128 else 0)

        if self.transformer:
            img, mask = self.transformer((img, mask))


        if isinstance(mask, torch.Tensor):
            mask = torch.clamp(mask, 0.0, 1.0)

        return img, mask, img_path

    def __len__(self):
        return len(self.data)

    def get_raw_sample(self, idx):
        img_path, mask_path = self.data[idx]
        raw_img = Image.open(img_path).convert('RGB')
        raw_mask = Image.open(mask_path).convert('L')
        return raw_img, raw_mask


def visualize_comparison(dataset, idx, config):
    """
 
 dataset: 
 idx: 
 config: 
    """

    raw_img, raw_mask = dataset.get_raw_sample(idx)
    raw_img_np = np.array(raw_img)
    raw_mask_np = np.array(raw_mask)


    proc_img, proc_mask, img_path = dataset[idx]


 print(f"\n[] #{idx}:")
 print(f": {proc_img.shape},: {type(proc_img)},: [{proc_img.min()}, {proc_img.max()}]")
 print(f": {proc_mask.shape},: {torch.unique(proc_mask)}")


    if config.name == "STDS":
        mean = [0.40740484, 0.43896195, 0.46943511]
        std = [0.2454432, 0.25077291, 0.25487435]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    if config.name == "MICCAI2023":
        mean = [0.40347644, 0.40347645, 0.40347646]
        std = [0.16715204, 0.16715206, 0.16715206]


    denorm_img = denormalize(proc_img, mean, std)


    if isinstance(proc_mask, torch.Tensor):
        # (H, W)
        if proc_mask.dim() == 4:  # (B, C, H, W)
            proc_mask_np = proc_mask.squeeze(0).squeeze(0).numpy()
        elif proc_mask.dim() == 3:  # (C, H, W) or (H, W, C)
            if proc_mask.size(0) in [1]:  # (1, H, W)
                proc_mask_np = proc_mask.squeeze(0).numpy()
            elif proc_mask.size(1) in [1]:  # (H, 1, W)
                proc_mask_np = proc_mask.squeeze(1).numpy()
            elif proc_mask.size(2) in [1]:  # (H, W, 1)
                proc_mask_np = proc_mask.squeeze(2).numpy()
            else:
                proc_mask_np = proc_mask.numpy()
        else:  # (H, W)
            proc_mask_np = proc_mask.numpy()

        # 0-255numpy
        proc_mask_np = (proc_mask_np * 255).astype(np.uint8)
    else:
        proc_mask_np = (proc_mask.squeeze() * 255).astype(np.uint8)


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Sample #{idx} | Split: {dataset.split.upper()} | Dataset: {config.name.upper()}', fontsize=16)


    axes[0, 0].imshow(raw_img_np)
    axes[0, 0].set_title(f'Original Image\nSize: {raw_img_np.shape[:2]}')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(raw_mask_np, cmap='gray')
    axes[0, 1].set_title(f'Original Mask\nUnique: {np.unique(raw_mask_np)}')
    axes[0, 1].axis('off')


    axes[0, 2].imshow(raw_img_np)
    axes[0, 2].imshow(raw_mask_np, alpha=0.5, cmap='jet')
    axes[0, 2].set_title('Original Overlay')
    axes[0, 2].axis('off')


    axes[1, 0].imshow(denorm_img)
    axes[1, 0].set_title(f'Preprocessed Image\nSize: {denorm_img.shape[:2]}')
    axes[1, 0].axis('off')


    axes[1, 1].imshow(proc_mask_np, cmap='gray')
    axes[1, 1].set_title(f'Preprocessed Mask\nUnique: {np.unique(proc_mask_np)}')
    axes[1, 1].axis('off')


    axes[1, 2].imshow(denorm_img)
    axes[1, 2].imshow(proc_mask_np, alpha=0.5, cmap='jet')
    axes[1, 2].set_title('Preprocessed Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'visualization_sample_{idx}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    data_config = DataConfig()
    dataset_path = data_config.root_dir


    full_dataset = XJQ_datasets(
        path_Data=dataset_path,
        config=data_config,
        split='full'  # Note: split parameter does not affect initial data loading
    )


    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size


    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # （）
    train_dataset.split = 'train'
    val_dataset.split = 'val'
    test_dataset.split = 'test'

    print("=" * 60)
 print(f":")
 print(f": {len(train_dataset)} ")
 print(f": {len(val_dataset)} ")
 print(f": {len(test_dataset)} ")
    print("=" * 60)


    for name, dataset in zip(['Train', 'Val', 'Test'], [train_dataset, val_dataset, test_dataset]):
 print(f"\n {name}:")
        for i in range(min(2, len(dataset))):  # 2
            idx = dataset.indices[i] if hasattr(dataset, 'indices') else i
            visualize_comparison(full_dataset, idx, data_config)
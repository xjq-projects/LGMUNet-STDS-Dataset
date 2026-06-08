from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from torchvision import transforms
from utils import *


@dataclass
class ModelConfig:
 """LGMUNet"""
    network: str = "LGMUNet"  # lgmunet
    model_type: str = "standard"  # standard/lightweight
    num_classes: int = 1
    input_channels: int = 3

    # LGMUNet
    patch_size: int = 8
    convolution_stem_down: int = 8  # Convolution stem downsampling factor
    encoder_depths: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    decoder_depths: List[int] = field(default_factory=lambda: [3, 3, 3])
    embed_dim: int = 96
    d_state: int = 16  # SSM
    drop_path_rate: float = 0.1
    drop_rate: float = 0.0
    deep_supervision: bool = False

    pretrained_path: Optional[str] = None
    patch_norm: bool = True
    use_checkpoint: bool = False

    # VMUNet（）
    # scan_directions: int = 4 #


@dataclass
class DataConfig:
 """"""
    name: str = "MICCAI2023"  # isic18/isic17
    root_dir: str = field(init=False)
    input_size: Tuple[int, int] = (448, 448)
    batch_size: int = 8
    num_workers: int = 4
    train_transforms: transforms.Compose = field(init=False)
    test_transforms: transforms.Compose = field(init=False)

    if name == "isic18":
        root_dir = "./data/data_isic1718/isic2018/"
    elif name == "isic17":
        root_dir = "./data/data_isic1718/isic2017/"
    elif name == "STDS":
        root_dir = "./data/STDS/"
    elif name == "STDS":
        root_dir = "./data/STDS/"
    elif name == "MICCAI2023":
        root_dir == "./data/MICCAI2023/"
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_transforms = transforms.Compose([
        myResize(448, 448),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        myGaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        myToTensor(),
        myNormalize('MICCAI2023', train=True)
    ])

    test_transforms = transforms.Compose([
        myResize(448, 448),
        myToTensor(),
        myNormalize('MICCAI2023', train=False)
    ])


@dataclass
class TrainingConfig:
 """"""
    epochs: int = 300
    amp: bool = False
    seed: int = 42
    gpu_ids: List[str] = field(default_factory=lambda: ["0"])
    checkpoint_dir: str = field(init=False)
    work_dir: str = field(init=False)  # （ checkpoint_dir）
    datasets: str = field(init=False)

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01  # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9  # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6  # default: 1e-6 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adagrad':
        lr = 0.01  # default: 0.01 – learning rate
        lr_decay = 0  # default: 0 – learning rate decay
        eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001  # default: 1e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001  # default: 1e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2  # default: 1e-2 – weight decay coefficient
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3  # default: 2e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01  # default: 1e-2 – learning rate
        lambd = 1e-4  # default: 1e-4 – decay term
        alpha = 0.75  # default: 0.75 – power for eta update
        t0 = 1e6  # default: 1e6 – point at which to start averaging
        weight_decay = 0  # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        momentum = 0  # default: 0 – momentum factor
        alpha = 0.99  # default: 0.99 – smoothing constant
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        etas = (0.5,
                1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01  # – learning rate
        momentum = 0.9  # default: 0 – momentum factor
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        dampening = 0  # default: 0 – dampening for momentum
        nesterov = False  # default: False – enables Nesterov momentum

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5  # – Period of learning rate decay.
        gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
        gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99  # – Multiplicative factor of learning rate decay.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50  # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50  # – Number of iterations for the first restart.
        T_mult = 2  # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20

    checkpoint_dir: str = field(init=False)
    work_dir: str = ''
    print_interval: int = 40
    val_interval: int = 10
    save_interval: int = 30
    threshold = 0.5

    def __post_init__(self):
        # （）
        if not hasattr(self, 'work_dir'):
            self.work_dir = self.checkpoint_dir

        if not hasattr(self, 'datasets'):
            self.datasets = "STDS"
        timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
        self.checkpoint_dir = f"results/{self.opt}_{self.sch}_{timestamp}/"




@dataclass
class LossConfig:
 """"""
    name: str = "BceDiceLoss"
    params: Dict[str, Any] = field(default_factory=lambda: {"wb": 1.2, "wd": 0.8})


@dataclass
class Config:
 """"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
        self.train.checkpoint_dir = (
            f"results/{self.model.network}_{self.data.name}_{timestamp}/"
        )
        self.train.work_dir = self.train.checkpoint_dir
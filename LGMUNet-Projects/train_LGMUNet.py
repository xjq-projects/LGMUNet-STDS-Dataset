import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import json
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm

# Custom module imports
from configs.Config_LGMUNet import Config
from models.LGFMUNet.LGMUNet import LGMUNet_Factory
from dataset import XJQ_datasets
from utils import get_logger, set_seed, BceDiceLoss, get_optimizer, get_scheduler, cal_params_flops
from engine_lvmunet import train_one_epoch, val_one_epoch, test_one_epoch


def create_lgmunet(config):
    """Create LGMUNet model instance"""
    return LGMUNet_Factory(
        input_channels=config.model.input_channels,
        num_classes=config.model.num_classes,
        encoder_depths=config.model.encoder_depths,
        decoder_depths=config.model.decoder_depths,
        embed_dim=config.model.embed_dim,
        d_state=config.model.d_state,
        drop_path_rate=config.model.drop_path_rate,
        load_ckpt_path=config.model.pretrained_path,
        deep_supervision=config.model.deep_supervision
    )


def setup_directories(config):
    """Create necessary directory structure."""
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(config.train.checkpoint_dir, 'logs')
    vis_dir = os.path.join(config.train.checkpoint_dir, 'visualizations')
    tensorboard_dir = os.path.join(config.train.checkpoint_dir, 'tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    return log_dir, vis_dir, tensorboard_dir


def main(config, seed=None):
    """Single-seed training main function. Returns best val and test metrics."""
    # Use the passed seed or config seed
    run_seed = seed if seed is not None else config.train.seed

    # Set random seed
    set_seed(run_seed)

    # Update config seed
    config.train.seed = run_seed

    # Create seed-specific checkpoint dir
    base_checkpoint_dir = config.train.checkpoint_dir
    seed_checkpoint_dir = os.path.join(base_checkpoint_dir, f'seed_{run_seed}')
    config.train.checkpoint_dir = seed_checkpoint_dir

    # Create directories
    log_dir, vis_dir, tensorboard_dir = setup_directories(config)
# Initialize logger and TensorBoard
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Log config info
    logger.info(f"Starting training with seed {run_seed} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {config.model.network}, Data: {config.data.name}")

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.train.gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ================== Data Preparation ==================
    logger.info("#----------Preparing dataset----------#")

    dataset_path = config.data.root_dir
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset directory {dataset_path} does not exist")

    # Create full dataset
    full_dataset = XJQ_datasets(
        path_Data=dataset_path,
        config=config.data,
        split='full'
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
# Randomly split dataset using current seed    torch.manual_seed(run_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
# Set dataset types    setattr(train_dataset, 'split', 'train')
    setattr(val_dataset, 'split', 'val')
    setattr(test_dataset, 'split', 'test')

    logger.info(f"Total dataset size: {total_size}")
    logger.info(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
# Create data loaders    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # ================== Data Preparation ==================
    logger.info("#----------Preparing Model----------#")

    model = create_lgmunet(config)

    if len(config.train.gpu_ids) > 1:
        logger.info(f"Using {len(config.train.gpu_ids)} GPUs for training")
        model = torch.nn.DataParallel(model, device_ids=list(range(len(config.train.gpu_ids))))

    model = model.to(device)
    cal_params_flops(model, config.data.input_size[0], logger)

    # ================== Data Preparation ==================
    logger.info("#----------Preparing loss, optimizer and scheduler----------#")

    criterion = BceDiceLoss(**config.loss.params)
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer)

    logger.info(f"Loss: {type(criterion).__name__}, Optimizer: {type(optimizer).__name__}, Scheduler: {type(scheduler).__name__}")

    # ================== Data Preparation ==================
    logger.info("#----------Starting Training----------#")

    start_epoch = 1
    best_dice = 0.0
    best_epoch = 0
    best_val_metrics = None
    step = 0

    config.train.work_dir = config.train.checkpoint_dir
    config.train.datasets = config.data.name

    for epoch in range(start_epoch, config.train.epochs + 1):
        logger.info(f"Epoch {epoch}/{config.train.epochs}")

        step = train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            logger=logger,
            config=config.train,
            writer=writer
        )

        # val_one_epoch
        val_metrics = val_one_epoch(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            logger=logger,
            config=config.train,
            writer=writer
        )
# Save best model based on Dice score
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            best_val_metrics = {k: v for k, v in val_metrics.items()}
            save_path = os.path.join(config.train.checkpoint_dir, 'best_model.pth')

            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

            logger.info(f"New best model at epoch {epoch} | DSC: {best_dice:.4f} | mIoU: {val_metrics['miou']:.4f}")
# Save latest checkpoint        checkpoint = {
            'epoch': epoch,
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step
        }
        torch.save(checkpoint, os.path.join(config.train.checkpoint_dir, 'latest.pth'))

    # ================== Data Preparation ==================
    logger.info("#----------Final Testing----------#")

    best_model_path = os.path.join(config.train.checkpoint_dir, 'best_model.pth')
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(best_model_path))
    else:
        model.load_state_dict(torch.load(best_model_path))

    logger.info(f"Loaded best model from epoch {best_epoch} (val DSC: {best_dice:.4f}) for testing")

    # test_one_epoch
    test_metrics = test_one_epoch(
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        logger=logger,
        config=config.train,
        writer=writer,
        test_data_name=config.data.name
    )

    writer.close()
# Build result dict    result = {
        'seed': run_seed,
        'best_epoch': best_epoch,
        'best_val': best_val_metrics,
        'test': test_metrics
    }

    return result


def print_summary_table(all_results, metric_names, title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Metric':<12} " + "".join(f"{'Run'+str(i+1):>10}" for i in range(len(all_results))) + f"  {'Mean':>10}  {'Std':>10}"
    print(header)
    print("-" * len(header))

    agg = {}
    for name in metric_names:
        vals = [r['test'][name] for r in all_results]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)  # (ddof=1)
        agg[name] = {'mean': float(mean), 'std': float(std), 'values': [float(v) for v in vals]}
        vals_str = "".join(f"{v:>10.4f}" for v in vals)
        print(f"{name:<12} {vals_str}  {mean:>10.4f}  {std:>10.4f}")
    print(f"{'='*80}\n")
    return agg


if __name__ == "__main__":
# Create config object    config = Config()
# Override default configuration    config.model.network = "lgmunet"
    config.model.model_type = "standard"
    config.data.name = "MICCAI2023"
    config.train.epochs = 300
    config.data.batch_size = 8
    config.data.root_dir = "./data/MICCAI2023/"
# LGMUNet-specific configuration
    config.model.encoder_depths = [3, 3, 3, 3]
    config.model.decoder_depths = [3, 3, 3]
    config.model.embed_dim = 96
    config.model.d_state = 16
    config.model.deep_supervision = False

    # ================== Data Preparation ==================
    SEEDS = [42, 3407, 1234]

    # （）
    base_timestamp = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
    base_result_dir = f"results/{config.model.network}_{config.data.name}_{base_timestamp}"
    config.train.checkpoint_dir = base_result_dir
    os.makedirs(base_result_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  Multi-Seed Training: {config.model.network.upper()}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Results dir: {base_result_dir}")
    print(f"{'#'*60}\n")

    all_results = []
    metric_names = ['loss', 'miou', 'dice', 'acc', 'sens', 'spec']

    for i, seed in enumerate(SEEDS):
        print(f"\n{'*'*60}")
        print(f"  SEED {i+1}/{len(SEEDS)}: {seed}")
        print(f"{'*'*60}")

        try:
            result = main(config, seed=seed)
            all_results.append(result)
# Print single-seed results            print(f"\n  Seed {seed} | Best epoch: {result['best_epoch']}")
            print(f"  Best val  -> mIoU: {result['best_val']['miou']:.4f} | DSC: {result['best_val']['dice']:.4f} | Acc: {result['best_val']['acc']:.4f}")
            print(f"  Test      -> mIoU: {result['test']['miou']:.4f} | DSC: {result['test']['dice']:.4f} | Acc: {result['test']['acc']:.4f}")

        except Exception as e:
            logging.error(f"Training failed for seed {seed}: {str(e)}")
            print(f"ERROR: Seed {seed} failed: {str(e)}")
            import traceback
            traceback.print_exc()

    if len(all_results) == 0:
        print("ERROR: All seeds failed!")
        exit(1)

    # ================== Data Preparation ==================
    print(f"\n{'#'*60}")
    print(f"  MULTI-SEED RESULTS SUMMARY")
    print(f"  Model: {config.model.network.upper()}")
    print(f"  Seeds completed: {len(all_results)}/{len(SEEDS)}")
    print(f"{'#'*60}")

    agg_results = print_summary_table(all_results, metric_names,
                                       f"Test Set Metrics (n={len(all_results)} seeds)")
# Also print best validation metrics summary    val_agg = {}
    print(f"\n{'='*80}")
    print(f"  Best Validation Metrics Summary")
    print(f"{'='*80}")
    header = f"{'Metric':<12} " + "".join(f"{'Run'+str(i+1):>10}" for i in range(len(all_results))) + f"  {'Mean':>10}  {'Std':>10}"
    print(header)
    print("-" * len(header))
    for name in metric_names:
        vals = [r['best_val'][name] for r in all_results]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        val_agg[name] = {'mean': float(mean), 'std': float(std), 'values': [float(v) for v in vals]}
        vals_str = "".join(f"{v:>10.4f}" for v in vals)
        print(f"{name:<12} {vals_str}  {mean:>10.4f}  {std:>10.4f}")

    # ================== JSON ==================
    results_data = {
        'model': config.model.network,
        'dataset': config.data.name,
        'seeds': SEEDS,
        'seeds_completed': len(all_results),
        'per_seed': [{
            'seed': r['seed'],
            'best_epoch': r['best_epoch'],
            'best_val': r['best_val'],
            'test': r['test']
        } for r in all_results],
        'aggregated_val': val_agg,
        'aggregated_test': agg_results
    }

    json_path = os.path.join(base_result_dir, 'multi_seed_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {json_path}")
    print("Multi-seed training completed!")

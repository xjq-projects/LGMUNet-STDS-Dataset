import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import os


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    # Switch to training mode
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()

        # Unpack data: images, targets, paths (ignore paths)
        images, targets, _ = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        # Forward pass（）
        with autocast(enabled=config.amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Record loss
        loss_list.append(loss.item())
        now_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/loss', loss.item(), global_step=step)

        # Log printing
        if iter % config.print_interval == 0:
            log_info = f'Train Epoch: {epoch} | Iter: {iter}/{len(train_loader)} | Loss: {np.mean(loss_list):.4f} | LR: {now_lr:.6f}'
            print(log_info)
            logger.info(log_info)

    # Learning rate scheduling
    scheduler.step()
    return step


def val_one_epoch(val_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config,
                  writer):
    # Switch to evaluation mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            # Unpack data: images, targets, paths (ignore paths)
            images, targets, _ = data
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True).float()

            # Forward inference
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss_list.append(loss.item())

            # Process output (adapt to multi-scale output)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            gts.append(targets.cpu().detach().numpy().reshape(-1))  # Flatten directly
            preds.append(outputs.cpu().detach().numpy().reshape(-1))

    # Compute average loss
    avg_loss = np.mean(loss_list)

    # Always compute detailed metrics
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)

    # Binarize predictions and ground truth
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    # Compute confusion matrix
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    # Compute all metrics
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    metrics = {
        'loss': avg_loss,
        'miou': miou,
        'dice': f1_or_dsc,
        'acc': accuracy,
        'sens': sensitivity,
        'spec': specificity
    }

    # Print mIoU and DSC per epoch
    log_info = f'Val Epoch {epoch} | Loss: {avg_loss:.4f} | mIoU: {miou:.4f} | DSC: {f1_or_dsc:.4f}'
    print(log_info)
    logger.info(log_info)

    # Print full metrics at validation interval
    if epoch % config.val_interval == 0:
        detail_info = (f'Val Epoch {epoch} Detail | Acc: {accuracy:.4f} | '
                       f'Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | '
                       f'TN:{TN} FP:{FP} FN:{FN} TP:{TP}')
        print(detail_info)
        logger.info(detail_info)

    # TensorBoard logging
    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/mIoU', miou, epoch)
        writer.add_scalar('val/f1_dsc', f1_or_dsc, epoch)
        writer.add_scalar('val/accuracy', accuracy, epoch)
        writer.add_scalar('val/specificity', specificity, epoch)
        writer.add_scalar('val/sensitivity', sensitivity, epoch)

    return metrics


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   writer=None,
                   test_data_name=None):
    model.eval()
    loss_list = []
    preds, gts = [], []

    # Ensure visualization directory exists
    vis_dir = os.path.join(config.work_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # Unpack data: images, targets, paths
            images, targets, img_paths = data
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True).float()

            # Forward inference
            with autocast(enabled=config.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss_list.append(loss.item())

            # Save visualization results
            if i % config.save_interval == 0:
                # Extract filenames for saving
                file_names = [os.path.basename(path) for path in img_paths]

                save_imgs(
                    images.cpu(),
                    targets.cpu(),
                    outputs.cpu(),
                    file_names,  # Use filename list
                    vis_dir,
                    config.datasets,
                    config.threshold,
                    test_data_name
                )

            # Collect predictions
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds.append(outputs.cpu().numpy())
            gts.append(targets.cpu().numpy())

    # Metrics computation
    preds = np.concatenate(preds).flatten()
    gts = np.concatenate(gts).flatten()
    y_pred = (preds >= config.threshold).astype(int)
    y_true = (gts >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'loss': np.mean(loss_list),
        'dice': (2 * tp) / (2 * tp + fp + fn + 1e-8),
        'miou': tp / (tp + fp + fn + 1e-8),  # Note: key name is 'miou'
        'acc': (tp + tn) / (tp + tn + fp + fn + 1e-8),
        'sens': tp / (tp + fn + 1e-8),
        'spec': tn / (tn + fp + 1e-8)
    }

    # Log test results
    log_info = (f"Test | Loss: {metrics['loss']:.4f} | "
                f"DSC/F1: {metrics['dice']:.4f} | mIoU: {metrics['miou']:.4f} | "
                f"Acc: {metrics['acc']:.4f} | Sens: {metrics['sens']:.4f} | Spec: {metrics['spec']:.4f}")
    if test_data_name:
        log_info = f"Dataset: {test_data_name} | " + log_info
    print(log_info)
    logger.info(log_info)

    return metrics
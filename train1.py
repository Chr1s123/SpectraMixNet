import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from matplotlib import pyplot as plt # Removed as plotting code was commented out
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
# from utils.CR import ContrastLoss # Assuming ContrastLoss_res is the one used
from utils.CR_res import ContrastLoss_res
# from skimage.metrics import structural_similarity as ssim # Removed as SSIM calculation was commented out
import numpy as np
import time  # <-- Added import
import datetime  # <-- Added import

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser()
parser.add_argument('--stop_epoch',  type=int, default=None,help='达到该 epoch（从 1 开始计）后退出训练')#None
parser.add_argument('--model', default='SpectraMixNet-t', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of   workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='../Dehazing/data/RESIDE-IN/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='../Dehazing/logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')  # Consider setting a default if needed
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler, setting, current_epoch):
    # Pass current_epoch for accurate tqdm description
    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.train()

    # Use tqdm for progress bar, including epoch number
    pbar = tqdm(train_loader, desc=f"Training Epoch {current_epoch + 1}/{setting['epochs']}")
    for i, batch in enumerate(pbar):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(enabled=args.no_autocast):  # Use enabled= for clarity
            output = network(source_img)
            loss = criterion[0](output, target_img) + criterion[1](output, target_img, source_img) * 0.1
            # ablation-base
            # loss = criterion[0](output, target_img)

        losses.update(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar description with current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{losses.avg:.4f}")

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    # SSIM = AverageMeter() # Removed as SSIM calculation was commented out
    torch.cuda.empty_cache()
    network.eval()

    pbar = tqdm(val_loader, desc="Validating", leave=False)  # Add progress bar for validation
    for batch in pbar:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            # Ensure output is clamped to [-1, 1] if input is normalized to [-1, 1]
            output = network(source_img).clamp_(-1, 1)

        # Calculate PSNR
        # Assuming target_img and output are in [-1, 1], scale to [0, 1] for PSNR calculation
        output_psnr = output * 0.5 + 0.5
        target_psnr = target_img * 0.5 + 0.5
        mse_loss = F.mse_loss(output_psnr, target_psnr, reduction='none').mean((1, 2, 3))
        # Handle potential issues with mse_loss being zero
        psnr = 10 * torch.log10(1 / (mse_loss + 1e-8)).mean()  # Add epsilon for stability
        PSNR.update(psnr.item(), source_img.size(0))
        pbar.set_postfix(avg_psnr=f"{PSNR.avg:.4f}")

    return PSNR.avg  # Return only PSNR


# Helper function to format time
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


if __name__ == '__main__':
    # ... (Keep the initial print statements and config loading) ...
    print("--- Training Configuration ---")
    print(f"Model: {args.model}")
    print(f"Experiment: {args.exp}")
    print(f"Dataset: {args.dataset if args.dataset else 'Default in data_dir'}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Use Autocast: {args.no_autocast}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Visible GPUs: {args.gpu}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("-----------------------------")

    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        print(f"Warning: Specific config '{setting_filename}' not found. Using default.")
        setting_filename = os.path.join('configs', args.exp, 'default.json')
        if not os.path.exists(setting_filename):
            raise FileNotFoundError(f"Default config '{setting_filename}' also not found!")

    with open(setting_filename, 'r') as f:
        setting = json.load(f)
        print(f"Loaded settings from: {setting_filename}")
        print("Settings:", json.dumps(setting, indent=2))

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    criterion = [nn.L1Loss().cuda(), ContrastLoss_res(ablation=False).cuda()]

    if setting['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception(f"ERROR: unsupported optimizer '{setting['optimizer']}'")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler(enabled=args.no_autocast)  # Pass enabled flag

    # --- Checkpoint Loading ---
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(save_dir, f"{args.model}_latest.pth")
    best_model_path = os.path.join(save_dir, f"{args.model}_best.pth")  # Separate name for best model

    start_epoch = 0
    best_psnr = 0.0

    if os.path.exists(latest_checkpoint_path):
        print(f"Resuming training from latest checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        try:
            network.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
            best_psnr = checkpoint.get('best_psnr', 0.0)  # Use .get for backward compatibility
            # Restore scheduler state *after* potential epoch change
            # scheduler.last_epoch = start_epoch - 1 # Adjust if needed based on scheduler type
            print(
                f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}. Best PSNR so far: {best_psnr:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_psnr = 0.0
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Data Loaders ---
    # ... (Keep data loader setup) ...
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    print(f"Loading dataset from: {dataset_dir}")
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'],
                               setting.get('edge_decay', 0),  # Use .get for safety
                               setting.get('only_h_flip', False))  # Use .get for safety
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,  # Shuffle is important for training
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting.get('patch_size',
                                         None))  # Use setting patch_size if defined, else None? Check PairLoader logic
    # setting['patch_size']) # Original: Assuming val uses same patch size
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            # Consider smaller batch size for validation if memory is an issue
                            shuffle=False,  # No need to shuffle validation data
                            num_workers=args.num_workers,
                            pin_memory=True)

    print(f"Training batch size: {setting['batch_size']}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # --- TensorBoard Writer ---
    log_path = os.path.join(args.log_dir, args.exp, args.model)
    print(f"Logging to TensorBoard: {log_path}")
    writer = SummaryWriter(log_dir=log_path)

    # --- Training Loop ---
    print(f"==> Starting training from epoch {start_epoch + 1} to {setting['epochs']}")
    training_start_time = time.time()  # <-- Record overall start time

    for epoch in range(start_epoch, setting['epochs']):
        if args.stop_epoch is not None and (epoch + 1) >= args.stop_epoch:
            print(f"\nReached stop_epoch={args.stop_epoch}, exiting training.")
            break
        epoch_start_time = time.time()  # <-- Record epoch start time

        # Train for one epoch
        # Pass epoch to train function for tqdm description
        train_loss = train(train_loader, network, criterion, optimizer, scaler, setting, epoch)

        # Validate for one epoch
        avg_psnr = valid(val_loader, network)

        # --- Calculate Times ---
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - training_start_time

        epochs_completed = epoch - start_epoch + 1
        total_epochs_in_run = setting['epochs'] - start_epoch  # Total epochs for this run

        # Estimate remaining time
        if epochs_completed > 0:
            average_epoch_time = total_elapsed_time / epochs_completed
            remaining_epochs = total_epochs_in_run - epochs_completed
            estimated_remaining_time = average_epoch_time * remaining_epochs
        else:  # Should not happen if loop runs at least once
            estimated_remaining_time = 0

        # --- Get current learning rate ---
        current_lr = optimizer.param_groups[0]['lr']

        # --- Print Epoch Summary ---
        print(
            f"Epoch [{epoch + 1}/{setting['epochs']}] | "
            f"Loss: {train_loss:.6f} | "
            f"PSNR: {avg_psnr:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {format_time(epoch_duration)} | "  # Epoch duration
            f"Elapsed: {format_time(total_elapsed_time)} | "  # Total elapsed
            f"ETA: {format_time(estimated_remaining_time)}"  # Estimated time remaining
        )

        # --- Log to TensorBoard ---
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Valid/PSNR', avg_psnr, epoch)
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        writer.add_scalar('Time/Epoch_Duration_Seconds', epoch_duration, epoch)  # Log duration

        # --- Step the scheduler ---
        scheduler.step()

        # --- Save Best Model Checkpoint ---
        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr
            print("=" * 80)
            print(f"🌟 [New Best] PSNR = {best_psnr:.4f} | Model saved at: {best_model_path}")
            print("=" * 80)

            # print(f"*** New best PSNR: {best_psnr:.4f}. Saving best model to {best_model_path} ***")
            torch.save({
                'epoch': epoch,
                'state_dict': network.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),  # Also save optimizer with best model
                'lr_scheduler': scheduler.state_dict(),  # And scheduler
                'scaler': scaler.state_dict(),  # And scaler state
            }, best_model_path)

        # --- Save Latest Checkpoint ---
        # Always save the latest state at the end of epoch for resuming
        # print(f"Saving latest checkpoint to {latest_checkpoint_path}") # Optional: less verbose
        torch.save({
            'epoch': epoch,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_psnr': best_psnr,  # Include best_psnr encountered so far
        }, latest_checkpoint_path)

    # --- Cleanup ---
    writer.close()
    total_training_time = time.time() - training_start_time
    print("==> Training finished.")
    print(f"Total Training Time: {format_time(total_training_time)}")  # Print total time
    print(f"Best validation PSNR achieved: {best_psnr:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Latest model state saved to: {latest_checkpoint_path}")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ptflops import get_model_complexity_info # Keep commented if not used
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import time # Import time for potential profiling
from tqdm import tqdm # Import tqdm for progress bar

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import * # Make sure all your models are imported here

parser = argparse.ArgumentParser()
# --- Keep your argument definitions ---
parser.add_argument('--model', default='ImprovedMixDehazeNet-t', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
# Important: save_dir and exp should match the training setup
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='Parent directory where models were saved')
parser.add_argument('--exp', default='indoor', type=str, help='Experiment name used during training')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN/', type=str, help='dataset name (folder under data_dir)')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for testing')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Function to load state dict, handles 'module.' prefix from DataParallel
# And checks if the loaded object is a checkpoint dictionary or just the state_dict
def load_state_dict_flexible(model_path):
    """Loads state dict, handling DataParallel prefix and checkpoint dict."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu') # Load to CPU first
    # Check if it's a checkpoint dictionary or just the state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Loaded state_dict from epoch {checkpoint.get('epoch', 'N/A')}")
    elif isinstance(checkpoint, dict): # Assuming the loaded object is the state_dict itself
        state_dict = checkpoint
    else:
         raise TypeError(f"Loaded object is not a dictionary or state_dict: {type(checkpoint)}")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove `module.` prefix
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    network.eval() # Set model to evaluation mode
    torch.cuda.empty_cache()

    # Create result directories if they don't exist
    img_result_dir = os.path.join(result_dir, 'imgs')
    os.makedirs(img_result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, 'temp_results.csv') # Temporary CSV name

    # Add a progress bar using tqdm
    pbar = tqdm(test_loader, desc=f"Testing {args.model} on {args.dataset}")
    total_time = 0

    try:
        with open(csv_path, 'w') as f_result:
             f_result.write('filename,psnr,ssim\n') # Write header

             with torch.no_grad(): # Disable gradient calculations
                for idx, batch in enumerate(pbar):
                    input_img = batch['source'].cuda()
                    target_img = batch['target'].cuda()
                    filename = batch['filename'][0]

                    start_time = time.time()
                    output = network(input_img).clamp_(-1, 1)
                    end_time = time.time()
                    total_time += (end_time - start_time)

                    # Convert [-1, 1] to [0, 1] for metrics and saving
                    output_01 = output.detach() * 0.5 + 0.5
                    target_01 = target_img.detach() * 0.5 + 0.5

                    # Calculate PSNR
                    mse = F.mse_loss(output_01, target_01)
                    psnr_val = 10 * torch.log10(1 / (mse + 1e-8)).item() # Add epsilon for stability

                    # Calculate SSIM (using pytorch-msssim)
                    # Ensure tensors are on the same device (CPU or GPU) if needed by ssim library
                    ssim_val = ssim(output_01, target_01, data_range=1.0, size_average=True).item() # Use size_average=True for batch avg

                    PSNR.update(psnr_val)
                    SSIM.update(ssim_val)

                    # Write results to CSV
                    f_result.write(f'{filename},{psnr_val:.2f},{ssim_val:.4f}\n')

                    # Save output image
                    # Ensure output is on CPU, converted correctly (CHW->HWC), and scaled [0, 255]
                    out_img_np = chw_to_hwc(output_01.cpu().squeeze(0).numpy())
                    write_img(os.path.join(img_result_dir, filename), out_img_np) # Assuming write_img handles scaling

                    # Update progress bar description
                    pbar.set_postfix(PSNR=f"{PSNR.avg:.2f}", SSIM=f"{SSIM.avg:.4f}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")
        # Clean up temporary file if error occurs
        if os.path.exists(csv_path):
            os.remove(csv_path)
        raise # Re-raise the exception

    finally:
        # Rename CSV file to include final scores only if the loop completed successfully
        if os.path.exists(csv_path):
            final_csv_path = os.path.join(result_dir, f'results_{args.model}_PSNR{PSNR.avg:.2f}_SSIM{SSIM.avg:.4f}.csv')
            try:
                os.rename(csv_path, final_csv_path)
                print(f"Results saved to {final_csv_path}")
            except OSError as e:
                print(f"Error renaming result file: {e}")
                print(f"Temporary results might be in {csv_path}")


    avg_inference_time = total_time / len(test_loader) if len(test_loader) > 0 else 0
    print(f"--- Test Summary ---")
    print(f"Average PSNR: {PSNR.avg:.2f}")
    print(f"Average SSIM: {SSIM.avg:.4f}")
    print(f"Average Inference Time per image: {avg_inference_time:.4f} seconds")
    print(f"Result images saved in: {img_result_dir}")


if __name__ == '__main__':
    print("--- Testing Configuration ---")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Model Load Directory (Parent): {args.save_dir}")
    print(f"Experiment Name: {args.exp}")
    print(f"Results Directory: {args.result_dir}")
    print(f"GPU: {args.gpu}")
    print("-----------------------------")

    # --- Construct the path to the BEST model checkpoint ---
    model_dir = os.path.join(args.save_dir, args.exp) # Directory where models for this exp are saved
    best_model_filename = f"{args.model}_best.pth" # The filename for the best model
    best_model_path = os.path.join(model_dir, best_model_filename)
    print(f"Attempting to load best model weights from: {best_model_path}")

    # --- Load Model ---
    try:
        # Instantiate the network structure
        network = eval(args.model.replace('-', '_'))() # Make sure model class name matches pattern

        # Load the state dict using the flexible loader
        state_dict = load_state_dict_flexible(best_model_path)
        network.load_state_dict(state_dict)
        print(f"Successfully loaded weights into {args.model}")

        network.cuda() # Move model to GPU

    except FileNotFoundError:
        print(f"Error: Best model checkpoint not found at {best_model_path}")
        print("Please ensure '--save_dir', '--exp', and '--model' arguments match the training setup.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        exit(1)

    # --- Optional: Model Complexity (keep commented if not needed) ---
    # try:
    #     macs, params = get_model_complexity_info(network, (3, 256, 256), as_strings=True, # Use a representative input size
    #                                             print_per_layer_stat=False, verbose=False) # Less verbose output
    #     print('{:<30}  {:<8}'.format('Computational complexity (MACs):', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters:', params))
    # except Exception as e:
    #     print(f"Could not calculate model complexity: {e}")


    # --- Prepare Dataset and DataLoader ---
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        exit(1)

    test_dataset = PairLoader(dataset_dir, 'test', 'test') # Ensure mode ('test') is correct
    test_loader = DataLoader(test_dataset,
                             batch_size=1, # Test one image at a time
                             shuffle=False, # No need to shuffle for testing
                             num_workers=args.num_workers,
                             pin_memory=True)

    print(f"Found {len(test_dataset)} images in the test set.")

    # --- Prepare Result Directory ---
    # Results will be saved under result_dir/dataset_name/model_name/
    result_dir_specific = os.path.join(args.result_dir, args.dataset.replace('/','_'), args.model) # Sanitize dataset name for path
    os.makedirs(result_dir_specific, exist_ok=True)
    print(f"Results will be saved in: {result_dir_specific}")

    # --- Run Testing ---
    print('==> Starting testing...')
    test(test_loader, network, result_dir_specific)
    print('==> Testing finished.')
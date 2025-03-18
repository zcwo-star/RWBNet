import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = '/path/to/weld_seam/dataset'  # Update this to your weld seam dataset path
address = "last"  # Test the last trained network

################################################################################
# Network parameters (Adapted for LDDCNN)
################################################################################
net_class = sn.LDDCNN  # Assuming you have an LDDCNN class defined in src.networks
net_params = {
    'in_channels': 1,    # Grayscale weld seam images
    'out_channels': 1,   # Denoised output
    'n_filters': 64,     # Base number of filters
    'depth': 5,          # Number of convolutional layers
    'dropout': 0.2,      # Dropout rate for regularization
    'kernel_size': 3,    # Typical kernel size for denoising
    'batch_norm': True,  # Batch normalization for better training
}

################################################################################
# Dataset parameters (Adapted for weld seam images)
################################################################################
dataset_class = ds.WeldSeamDataset  # Custom dataset class for weld seam images
dataset_params = {
    'data_dir': data_dir,  # Directory containing noisy weld seam images
    'predata_dir': os.path.join(base_dir, 'data/weld_seam'),
    # Train/val/test split (example dataset names)
    'train_seqs': [
        'weld_batch_01',
        'weld_batch_02',
        'weld_batch_03',
    ],
    'val_seqs': [
        'weld_batch_04',
    ],
    'test_seqs': [
        'weld_batch_05',
        'weld_batch_06',
    ],
    # Image patch parameters
    'patch_size': 64,    # Size of image patches for training
    'stride': 32,        # Stride for patch extraction
    'augment': True,     # Data augmentation (rotation, flip, etc.)
}

################################################################################
# Training parameters (Adapted for denoising)
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.001,         # Lower learning rate for fine details
        'weight_decay': 1e-4,
        'amsgrad': True,
    },
    'loss_class': sl.DenoisingLoss,  # Custom loss for denoising (e.g., MSE + perceptual)
    'loss': {
        'mse_weight': 1.0,   # Weight for Mean Squared Error
        'perceptual_weight': 0.1,  # Weight for perceptual loss (if used)
    },
    'scheduler_class': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 50,
        'min_lr': 1e-6,
    },
    'dataloader': {
        'batch_size': 16,    # Adjust based on GPU memory
        'pin_memory': True,
        'num_workers': 4,    # Parallel data loading
        'shuffle': True,
    },
    'freq_val': 200,        # Validation frequency
    'n_epochs': 1000,       # Total epochs
    'res_dir': os.path.join(base_dir, "results/weld_seam_denoising"),
    'tb_dir': os.path.join(base_dir, "results/runs/weld_seam_denoising"),
}

################################################################################
# Train on weld seam dataset (Uncomment to train)
################################################################################
# learning_process = lr.DenoisingLearningBasedProcessing(
#     train_params['res_dir'],
#     train_params['tb_dir'],
#     net_class,
#     net_params,
#     None
# )
# learning_process.train(dataset_class, dataset_params, train_params)

################################################################################
# Test on weld seam dataset
################################################################################
learning_process = lr.DenoisingLearningBasedProcessing(
    train_params['res_dir'],
    train_params['tb_dir'],
    net_class,
    net_params,
    address=address
)
learning_process.test(dataset_class, dataset_params, ['test'])
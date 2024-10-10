import torch
from phenaki_pytorch import CViViT, CViViTTrainer
from accelerate import Accelerator

# Define the configuration for Accelerator
accelerate_kwargs = {
    'mixed_precision': 'fp16',  # use mixed precision training
    'split_batches': True
}

# Initialize the CViViT model
cvivit = CViViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 128,
    patch_size = 32,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 64,
    heads = 8
).cuda()

# Use Accelerator for model and data preparation
# accelerator = Accelerator(**accelerate_kwargs)
# cvivit, = accelerator.prepare(cvivit)

# Initialize the trainer
trainer = CViViTTrainer(
    vae=cvivit,  # Pass the unwrapped model
    folder='',
    batch_size=128,
    num_frames=11,
    grad_accum_every=4,
    train_on_images=False,
    use_ema=True,
    num_train_steps=1000000,
    save_model_every=5000,
    accelerate_kwargs=accelerate_kwargs
)

# Start training
trainer.train()  # Reconstructions and checkpoints will be saved periodically to ./results

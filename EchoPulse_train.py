import torch
from EchoPulse_pytorch.cvivit import CViViT
from EchoPulse_pytorch.cvivit_trainer import CViViTTrainer
from EchoPulse_pytorch.phenaki_pytorch import MaskGit, Phenaki
from EchoPulse_pytorch.phenaki_trainer import PhenakiTrainer

cvivit = CViViT(
    dim=512,  # embedding size
    codebook_size=8192,  # codebook size
    image_size=128,  # H,W
    patch_size=8,  # spatial patch size
    local_vgg=True,
    wandb_mode='disabled',
    temporal_patch_size=2,  # temporal patch size
    spatial_depth=4,  # nb of layers in the spatial transfo
    temporal_depth=4,  # nb of layers in the temporal transfo
    dim_head=64,  # hidden size in transfo
    heads=8,  # nb of heads for multi head transfo
    ff_mult=4,  # 32 * 64 = 2048 MLP size in transfo out
    commit_loss_w=1.,  # commit loss weight
    gen_loss_w=1.,  # generator loss weight
    perceptual_loss_w=1.,  # vgg loss weight
    i3d_loss_w=1.,  # i3d loss weight
    recon_loss_w=10.,  # reconstruction loss weight
    use_discr=0,  # whether to use a stylegan loss or not
    gp_weight=10
    
)


cvivit.load('pytorch_model.bin')

maskgit = MaskGit(
    num_tokens = 8192,  #set the same number with the codebook_size
    max_seq_len = 2048,
    dim = 512,
    dim_context = 768,
    depth = 6,
)

phenaki = Phenaki(
    cvivit = cvivit,
    maskgit = maskgit
).cuda()

trainer = PhenakiTrainer(
    phenaki = phenaki,
    folder =  '',
    train_on_images = False,
    batch_size = 256,
    grad_accum_every = 1,
    num_frames = 11,
    sample_num_frames = None,
    train_lr = 1e-4,
    train_num_steps = 4000000,
    max_grad_norm = None,
    ema_update_every = 10,
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    wd = 0,
    save_and_sample_every = 5000,
    num_samples = 25,
    results_folder = './phenaki_results',
    # amp = True,
    fp16 = True,
    split_batches = True,
    convert_image_to = None,
    sample_texts_file_path = '',  # path to a text file with video captions, delimited by newline
    losses_file_folder = './phenaki_results',
)

trainer.train()
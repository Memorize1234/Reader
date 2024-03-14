custom_imports = dict(imports=['model.encoders', 'model.decoders'], allow_failed_imports=False)

img_size = 256
encoder_only=False
num_group = 256

model = dict(
    g_model = dict(
        encoder_only=encoder_only,
        encoder = dict(
            type = 'ReaderEncoder',
            encoder_only = encoder_only,
            backbone = dict(
                img_size = img_size, 
                patch_size = 8, 
                embed_dim = 192, 
                depth = 12, 
                num_heads = 3),
            mask_decoder=dict(
                input_dim = 192,
                embed_dim = 256, 
                depth = 6, 
                num_heads = 8,
                num_queries = num_group,
                get_internm = False,
                assign_merge=True,
                use_mlp_mix=False,
                rgb_cfg = dict(
                    img_size = 256,
                    patch_size = 4,
                    embed_dim = 48)),
            mask_embed = dict(
                img_size = img_size, 
                patch_size = 4, 
                mask_size = 64, 
                rgb_dim = 48)),
        decoder = dict(
            type = 'ReaderDecoder',
            patch_size = 8,
            transformer = dict(
                latent_dim = 256,
                embed_dim = 192, 
                depth = 12, 
                num_heads = 3,
                num_queries = 1024),
            lpips_loss = dict(
                weight = 1.0,
                ckpt='checkpoints/vgg/vgg_lpips.pth'),
            l1_loss = dict(
                weight = 2.0))))

data = dict(
    train = dict(
        ann_root = 'path/to/sam_pseudo_anns',
        img_root = 'path/to/imagenet/train',
        pipeline_cfg = dict(
            img_size = img_size,
            mask_size_factor=1/4,
            need_anns = True,
            norm_type = 'standard')),
    train_loader = dict(
        batch_size = 32, # 512 in total for 16 GPUs
        num_workers = 4),
    val = dict(
        img_root = 'path/to/imagenet/val',
        img_sets = 'path/to/imagenet/val/val_filelist.txt',
        img_ids = [1, 2, 3, 9, 13])) # used for visualization during training)

train_cfg = dict(
    lr_cfg=dict(
        warm_up_cfg = dict(epoch = 1),
        type = 'default',
        lr = 2e-4,
        weight_decay = 0),
    do_validate = False,
    log_interval = 50,
    save_interval = 1)


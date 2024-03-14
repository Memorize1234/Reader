from torch import nn

def get_train_param(model_without_ddp, lr_cfg, pretrained_params, is_d=False):
    param_dict_type = lr_cfg.get('type', 'default')

    named_params = [(n, p) for n, p in model_without_ddp.named_parameters() if p.requires_grad]

    if param_dict_type == 'default':
        lr = lr_cfg.get('lr')
        train_params = [p for _, p in named_params]
    
    elif param_dict_type == 'frozen_part':
        train_params = []
        unused_params = set(lr_cfg['frozen_params'])
        for n, p in named_params:
            if n not in unused_params:
                train_params.append(p)
    
    elif param_dict_type == 'pretrained':
        lr, mult = lr_cfg.get('lr'), lr_cfg.get('mult')
        if pretrained_params is None:
            train_params = [p for _, p in named_params] 
        else:
            trained_params = []
            other_params = []
            for n, p in named_params:
                if 'g_model.' + n in pretrained_params:
                    trained_params.append(p)
                else:
                    other_params.append(p)
            train_params = [
                {"params": other_params},
                {"params": trained_params, "lr": lr * mult}]
    
    elif param_dict_type == 'finetune_encoder':
        lr, mult = lr_cfg.get('lr'), lr_cfg.get('mult')
        trained_params = []
        other_params = []
        for n, p in named_params:
            if 'encoder.' in n:
                trained_params.append(p)
            else:
                other_params.append(p)
        train_params = [
            {"params": other_params},
            {"params": trained_params, "lr": lr * mult}]
    
    elif param_dict_type == 'd_multi':
        lr, mult = lr_cfg.get('lr'), lr_cfg.get('mult')
        if is_d:
            lr = lr * mult
        train_params = [p for _, p in named_params]
    
    else:
        raise NotImplementedError(f"not implemented model params type{param_dict_type}")
    
    return train_params, lr

def get_named_params(named_params, name):
    res = []
    for n, p in named_params:
        if name in n:
            res.append(p)
    return res


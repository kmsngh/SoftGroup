import torch

def realign_parameter_keys(model_dict, ckpt_dict):
    model_keys = sorted(list(model_dict.keys()))
    ckpt_keys = sorted(list(ckpt_dict.keys()))
    
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in model_keys for j in ckpt_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(model_keys), len(ckpt_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1

    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            print("+++" * 5 + "{} not loaded".format(model_keys[idx_new]))
            continue
        key = model_keys[idx_new]
        key_old = ckpt_keys[idx_old]
        model_dict[key] = ckpt_dict[key_old]
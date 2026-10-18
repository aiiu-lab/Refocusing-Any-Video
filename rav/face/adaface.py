"""AdaFace loading and normalization helpers for optional face inference."""

import torch
from . import net_adaface

adaface_models = {
    'ir_50': './checkpoints/adaface/adaface_ir50_webface4m.ckpt',
}

def load_pretrained_model_adaface(architecture):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net_adaface.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model



def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


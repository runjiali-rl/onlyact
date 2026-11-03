# Sampling helpers from AdaWorld (worldmodel/sample_utils.py, Apache-2.0); only the functions used by generate.py are kept.
import torch

from vwm.modules.diffusionmodules.sampling import EulerEDMSampler


def get_guider(guider, scale):
    if guider == "IdentityGuider":
        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif guider == "VanillaCFG":
        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": scale
            }
        }
    else:
        raise ValueError(f"Unknown guider {guider}")
    return guider_config


def init_sampling(
        steps: int = 40,
        sampler: str = "EulerEDMSampler",
        discretization: str = "EDMShiftDiscretization",
        guider: str = "VanillaCFG",
        cfg_scale: float = 7.5,
        n_context_frames: int = 5
):
    discretization_config = get_discretization(discretization)
    guider_config = get_guider(guider, cfg_scale)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config, n_context_frames)
    return sampler


def get_discretization(discretization):
    if discretization == "EDMShiftDiscretization":
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMShiftDiscretization"
        }
    else:
        raise ValueError(f"Unknown discretization {discretization}")
    return discretization_config


def get_sampler(sampler, steps, discretization_config, guider_config, n_context_frames):
    if sampler == "EulerEDMSampler":
        s_churn = 0.0
        s_tmin = 0.0
        s_tmax = 999.0
        s_noise = 1.0

        sampler = EulerEDMSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            verbose=False,
            n_context_frames=n_context_frames
        )
    else:
        raise ValueError(f"Unknown sampler {sampler}")
    return sampler


def get_batch(keys, value_dict):
    batch = {}
    batch_uc = {}

    for key in keys:
        batch[key] = value_dict[key]
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

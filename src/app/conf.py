import os
from numba.core import typeinfer

import pydantic
import torch
import typing


class Base(pydantic.BaseSettings):
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    CLOUDINARY_CLOUD_NAME: str = pydantic.Field(..., env="CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = pydantic.Field(..., env="CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = pydantic.Field(..., env="CLOUDINARY_API_SECRET")
    
    # from main_end2end.py parser
    default_head_name: str = 'dali'
    ADD_NAIVE_EYE: bool = True
    CLOSE_INPUT_FACE_MOUTH: bool = False

    jpg: str = '{}.jpg'.format(default_head_name)
    close_input_face_mouth: bool = False # CLOSE_INPUT_FACE_MOUTH

    load_AUTOVC_name: str = 'examples/ckpt/ckpt_autovc.pth'
    load_a2l_G_name: str = 'examples/ckpt/ckpt_speaker_branch.pth'
    load_a2l_C_name: str = 'examples/ckpt/ckpt_content_branch.pth'
    load_G_name: str = 'examples/ckpt/ckpt_116_i2i_comb.pth'

    amp_lip_x: float = 2.
    amp_lip_y: float = 2.
    amp_pos: float = .5
    reuse_train_emb_list: typing.List[str] = []
    add_audio_in : bool = False
    comb_fan_awing : bool = False
    output_folder: str = 'examples'

    test_end2end : bool = True
    dump_dir: str = ''
    pos_dim : int = 7
    use_prior_net : bool = True
    transformer_d_model : int = 32
    transformer_N : int = 2
    transformer_heads : int = 2
    spk_emb_enc_size : int = 16
    init_content_encoder: str = ''
    lr: float = 1e-3 # learning rate
    reg_lr: float = 1e-6 # weight decay
    write : bool = False
    segment_batch_size: int = 1 # 'batch size'
    emb_coef : float = 3.0
    lambda_laplacian_smooth_loss : float = 1.0
    use_11spk_only : bool = False


# class Test(Base):
    # DEVICE: str = "cpu"


class Config(pydantic.BaseSettings):
    SETTINGS: pydantic.PyObject = "app.conf.Base"


config = Config()
settings = config.SETTINGS()

print(f"SETTINGS: {settings}")
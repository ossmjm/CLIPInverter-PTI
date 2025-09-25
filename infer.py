import argparse
import torch
from argparse import Namespace
import torchvision.transforms as transforms
import clip
import numpy as np
import sys
import os
import copy
import tempfile
import shutil
sys.path.append(".")
sys.path.append("..")
from models.e4e_features import pSp
from adapter.adapter_decoder import CLIPAdapterWithDecoder
from PIL import Image
from configs import global_config, hyperparameters
from run_pti import run_PTI
from utils.models_utils import toogle_grad
from models.stylegan2.model_remapper import Generator  # Added for allowlisting

def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')

def run_alignment(image_path):
    import dlib
    from align_faces_parallel import align_face
    predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(image_path, predictor=predictor)
    return aligned_image

def load_model(model_path, e4e_path):
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['pretrained_e4e_path'] = e4e_path
    opts['is_training_from_stage_one'] = False
    opts['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    opts = Namespace(**opts)
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(opts.device)

    adapter = CLIPAdapterWithDecoder(opts)
    adapter.eval()
    adapter.to(opts.device)

    clip_model, _ = clip.load("ViT-B/32", device=opts.device)
    return encoder, adapter, clip_model, opts.device

def manipulate(input_image_path, caption, encoder, adapter, clip_model, device, use_pti=False, neutral_prompt="a face"):
    input_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    input_real_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])])
    input_image_pil = Image.open(input_image_path).convert('RGB')
    aligned_image = run_alignment(input_image_pil)
    input_image_pil = input_real_transforms(input_image_pil).unsqueeze(0).to(device).float()
    input_image = input_transforms(aligned_image).unsqueeze(0).to(device).float()
    image_name = os.path.splitext(os.path.basename(input_image_path))[0]

    text_input = clip.tokenize(caption).to(device)
    text_features = clip_model.encode_text(text_input).float()

    with torch.no_grad():
        w, features = encoder(input_image, return_latents=True, randomize_noise=False)

    if use_pti:
        temp_embedding_dir = tempfile.mkdtemp()
        temp_checkpoints_dir = tempfile.mkdtemp()

        hyperparameters.first_inv_type = 'w+'

        run_name = run_PTI(run_name='pti_inference', use_wandb=False, use_multi_id_training=False,
                           preloaded_G=copy.deepcopy(adapter.decoder), preloaded_e4e=encoder,
                           clip_model=clip_model, neutral_prompt=neutral_prompt,
                           input_image=input_image_pil, image_name=image_name,
                           embedding_dir=temp_embedding_dir, checkpoints_dir=temp_checkpoints_dir,
                           initial_w=w)

        tuned_model_path = os.path.join(temp_checkpoints_dir, f'model_{run_name}_{image_name}.pt')
        torch.serialization.add_safe_globals([Generator, CLIPAdapterWithDecoder])  # Allowlist both classes
        tuned_state = torch.load(tuned_model_path, map_location=device)
        adapter.decoder.load_state_dict(tuned_state, strict=True)

        shutil.rmtree(temp_embedding_dir)
        shutil.rmtree(temp_checkpoints_dir)

    with torch.no_grad():
        features = adapter.adapter(features, text_features)
        delta = encoder.forward_features(features)
        w_hat = w + 0.1 * delta

    toogle_grad(adapter.decoder, False)
    result_tensor, _ = adapter.decoder([w_hat], input_is_latent=True, return_latents=False,
                                       randomize_noise=False, truncation=1, txt_embed=text_features)
    result_tensor = result_tensor.squeeze(0)
    result_image = tensor2im(result_tensor)
    return Image.fromarray(result_image)

def main(args):
    encoder, adapter, clip_model, device = load_model(args.model_path, args.e4e_path)
    result_image = manipulate(args.input_image_path, args.caption, encoder, adapter, clip_model, device,
                              use_pti=args.use_pti, neutral_prompt=args.neutral_prompt)
    os.makedirs("results", exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.input_image_path))[0]
    result_image.save(f"results/result_{filename}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="pretrained_models/pretrained_faces.pt")
    parser.add_argument("--e4e_path", type=str, default="pretrained_models/e4e_ffhq_encode.pt")
    parser.add_argument("--use_pti", action="store_true")
    parser.add_argument("--neutral_prompt", type=str, default="a face")
    args = parser.parse_args()
    main(args)
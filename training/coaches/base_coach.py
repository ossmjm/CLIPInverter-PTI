import abc
import os
import torch
from torchvision import transforms
from lpips import LPIPS
from configs import global_config, hyperparameters
from criteria import l2_loss
from criteria.localitly_regulizer import Space_Regulizer
import copy
from utils.models_utils import toogle_grad

class BaseCoach:
    def __init__(self, input_image, image_name, use_wandb, preloaded_G, preloaded_e4e, neutral_txt_features):
        self.use_wandb = use_wandb
        self.input_image = input_image
        self.image_name = image_name
        self.G = preloaded_G
        self.original_G = copy.deepcopy(preloaded_G)
        self.e4e_inversion_net = preloaded_e4e
        self.neutral_txt_features = neutral_txt_features

        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        toogle_grad(self.G, True)
        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def restart_training(self):
        toogle_grad(self.G, True)

    def get_inversion(self, w_path_dir, image_name, image, initial_w=None):
        embedding_dir = f'{w_path_dir}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = initial_w if initial_w is not None else self.get_e4e_inversion(image)
        torch.save(w_pivot, f'{embedding_dir}/0.pt')
        return w_pivot.to(global_config.device)

    def calc_inversions(self, image, image_name, initial_w=None):
        return initial_w if initial_w is not None else self.get_e4e_inversion(image)

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb, txt_embed=self.neutral_txt_features)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        if self.neutral_txt_features is not None:
            generated_images, _ = self.G([w], input_is_latent=True, return_latents=False, randomize_noise=False, truncation=1, txt_embed=self.neutral_txt_features)
            generated_images = generated_images.squeeze(0) if generated_images.dim() > 3 else generated_images
        else:
            generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        return generated_images

    def initilize_e4e(self):
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net.to(global_config.device)
        toogle_grad(self.e4e_inversion_net, False)

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])(image[0]).to(global_config.device)
        _, w = self.e4e_inversion_net(new_image.unsqueeze(0), randomize_noise=False, return_latents=True, resize=False,
                                      input_code=False)
        return w
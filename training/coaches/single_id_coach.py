import os
import torch
from tqdm import tqdm
from configs import hyperparameters, global_config
from training.coaches.base_coach import BaseCoach

class SingleIDCoach(BaseCoach):
    def __init__(self, input_image_inversion, input_image_loss, image_name, use_wandb, embedding_dir_path, checkpoints_dir, preloaded_G, preloaded_e4e, neutral_txt_features):
        self.embedding_dir_path = embedding_dir_path
        self.checkpoints_dir = checkpoints_dir
        super().__init__(input_image_inversion,input_image_loss, image_name, use_wandb, preloaded_G, preloaded_e4e, neutral_txt_features)

    def train(self, initial_w=None):
        use_ball_holder = True

        self.restart_training()

        embedding_dir = f'{self.embedding_dir_path}/{self.image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = self.get_inversion(self.embedding_dir_path, self.image_name, self.input_image_inversion, initial_w)
        w_pivot = w_pivot.to(global_config.device)

        torch.save(w_pivot, f'{embedding_dir}/0.pt')
        real_images_batch = self.input_image_loss.to(global_config.device)

        # print(f"[DEBUG] w_pivot requires_grad: {w_pivot.requires_grad}")
        # print(f"[DEBUG] real_images_batch requires_grad: {real_images_batch.requires_grad}")
        with torch.autograd.detect_anomaly():
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot).unsqueeze(0)
                #print(f"[DEBUG] Step {i}: generated_images requires_grad: {generated_images.requires_grad}")

                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, self.image_name,
                                                               self.G, use_ball_holder, w_pivot)
                if i % 20 == 0:
                    print(f"Step {i}: loss: {loss.item()}, l2_loss_val: {l2_loss_val.item()}, loss_lpips: {loss_lpips.item()}")

                #print(f"[DEBUG] Step {i}: loss requires_grad: {loss.requires_grad}, l2_loss_val: {l2_loss_val.item()}, loss_lpips: {loss_lpips.item()}")

                self.optimizer.zero_grad()

                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward(retain_graph=True)  # Temporary to debug
                l2_val = l2_loss_val.item()
                lpips_val = loss_lpips.item()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1

        torch.save(self.G.state_dict(), f'{self.checkpoints_dir}/model_{global_config.run_name}_{self.image_name}.pt')

        # produce final image in eval mode, no grads
        self.G.eval()
        with torch.no_grad():
            final_img = self.forward(w_pivot).unsqueeze(0).detach().cpu()

        return final_img
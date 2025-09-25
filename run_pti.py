from random import choice
from string import ascii_uppercase
import os
from configs import global_config, hyperparameters
from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
import clip
def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False, preloaded_G=None, preloaded_e4e=None, clip_model=None, neutral_prompt=None, input_image=None, image_name=None, embedding_dir=None, checkpoints_dir=None, initial_w=None):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    global_config.pivotal_training_steps = 1
    global_config.input_data_id = 'temp_input'

    if clip_model is not None and neutral_prompt is not None:
        neutral_text = clip.tokenize(neutral_prompt).to(global_config.device)
        neutral_txt_features = clip_model.encode_text(neutral_text).float()
    else:
        neutral_txt_features = None

    embedding_dir_path = f'{embedding_dir}/{global_config.input_data_id}/PTI'
    os.makedirs(embedding_dir_path, exist_ok=True)

    if use_multi_id_training:
        coach = MultiIDCoach(None, use_wandb)  # DataLoader not needed
    else:
        coach = SingleIDCoach(input_image, image_name, use_wandb, embedding_dir_path, checkpoints_dir, preloaded_G, preloaded_e4e, neutral_txt_features)
    result_image = coach.train(initial_w)

    return global_config.run_name, result_image

if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
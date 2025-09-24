## Architechture
lpips_type = 'alex'
first_inv_type = 'w+'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = True
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 10

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Steps
LPIPS_value_threshold = 0.05
max_pti_steps = 100

## Optimization
pti_learning_rate = 3e-4
use_last_w_pivots = False
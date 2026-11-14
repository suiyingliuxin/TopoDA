_base_ = ['topoda_crowdpose_r50.py']
# Small network for end-to-end regression checks, not a benchmark configuration.
enc_layers = 1
dec_layers = 3
num_box_decoder_layers = 1
dim_feedforward = 512
num_queries = 30
num_group = 5
num_select = 5
dn_number = 10
use_ema = False
data_aug_scales = [128]
data_aug_max_size = 192
data_aug_scales2_resize = [160]
data_aug_scales2_crop = [100, 140]
epochs = 2

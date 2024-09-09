# train a miniature character-level enwik8 model
# good for debugging and playing on macbooks and such

out_dir = "out-enwik8-char"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "enwik8"
wandb_run_name = "mini-gpt"

# these make the total batch size be ~60k
# 12 batch size * 1024 block size * 5 gradaccum= 61,440
# enwik8 has 90M tokens, so 1 epoch ~= 1500 iters

dataset = "enwik8_char"
gradient_accumulation_steps = 5
batch_size = 12
block_size = 1024  # context of up to 1024 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 1500
lr_decay_iters = 1500  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

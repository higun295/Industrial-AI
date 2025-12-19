from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile('configs/liif/liif_edsr_x4_300k_div2k.py')

# 1. Step 1: Pre-training (DIV2K)
cfg.train_dataloader.dataset.dataset.data_root = 'data/DIV2K'
cfg.train_cfg.max_iters = 300000
cfg.optim_wrapper.optimizer.lr = 1e-4
runner_pre = Runner.from_cfg(cfg)
runner_pre.train()

# 2. Step 2: Fine-tuning (AID/UCM)
cfg.load_from = 'work_dirs/liif/iter_300000.pth'
cfg.train_dataloader.dataset.dataset.data_root = 'data/AID'
cfg.train_cfg.max_iters = 100000
cfg.optim_wrapper.optimizer.lr = 5e-5
runner_fine = Runner.from_cfg(cfg)
runner_fine.train()

# 3. Step 3: Test
runner_fine.test()
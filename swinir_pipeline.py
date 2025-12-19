from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile('configs/swinir/swinir_x4s126w7_8xb1_500k_div2k.py')

# 1. Step 1: Pre-training (DIV2K)
cfg.train_dataloader.dataset.data_root = 'data/DIV2K'
cfg.train_cfg.max_iters = 500000 # SwinIR 특화 설정
cfg.optim_wrapper.optimizer.lr = 2e-4 # SwinIR 특화 설정
runner_pre = Runner.from_cfg(cfg)
runner_pre.train()

# 2. Step 2: Fine-tuning (AID/UCM)
cfg.load_from = 'work_dirs/swinir/iter_500000.pth'
cfg.train_dataloader.dataset.data_root = 'data/AID'
cfg.train_cfg.max_iters = 100000
cfg.optim_wrapper.optimizer.lr = 5e-5
runner_fine = Runner.from_cfg(cfg)
runner_fine.train()

# 3. Step 3: Test
runner_fine.test()
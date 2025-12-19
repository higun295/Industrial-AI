from mmagic.apis import MMagicInferencer
from mmengine.runner import Runner
from mmengine.config import Config

# 1. Configuration 로드 (EDSR 기본 설정)
cfg = Config.fromfile('configs/edsr/edsr_x4c64b16_1xb16_300k_div2k.py')

# 2. Step 1: Pre-training (DIV2K)
cfg.train_dataloader.dataset.dataset.data_root = 'data/DIV2K'
cfg.train_cfg.max_iters = 300000
cfg.optim_wrapper.optimizer.lr = 1e-4
runner_pre = Runner.from_cfg(cfg)
runner_pre.train()

# 3. Step 2: Fine-tuning (AID/UCM)
cfg.train_dataloader.dataset.dataset.data_root = 'data/AID' # 또는 UCM
cfg.load_from = 'work_dirs/edsr/iter_300000.pth' # 사전학습 가중치 로드
cfg.train_cfg.max_iters = 100000
cfg.optim_wrapper.optimizer.lr = 5e-5
runner_fine = Runner.from_cfg(cfg)
runner_fine.train()

# 4. Step 3: Test
runner_fine.test()
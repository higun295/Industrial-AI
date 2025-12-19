from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile('configs/real_esrgan/realesrnet_x4psnr_900k_iters.py')

# 1. Step 1: Pre-training (Real-ESRNet 단계)
cfg.train_dataloader.dataset.data_root = 'data/DIV2K'
cfg.train_cfg.max_iters = 300000
cfg.optim_wrapper.optimizer.lr = 1e-4
runner_pre = Runner.from_cfg(cfg)
runner_pre.train()

# 2. Step 2: Fine-tuning (Real-ESRGAN GAN 단계)
# GAN 설정으로 변경 (realesrgan_x4_600k_iters.py 내용 참조)
cfg.load_from = 'work_dirs/real_esrnet/iter_300000.pth'
cfg.train_dataloader.dataset.data_root = 'data/AID'
cfg.train_cfg.max_iters = 100000
cfg.optim_wrapper.optimizer.lr = 5e-5
runner_fine = Runner.from_cfg(cfg)
runner_fine.train()

# 3. Step 3: Test (PSNR, SSIM, LPIPS 측정)
runner_fine.test()
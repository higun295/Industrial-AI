# 🧠 Reinforcement Learning Project - Environment Setup Guide

This document describes how to set up a clean, stable Python environment for the Pong DQN reinforcement learning project using Anaconda and `pip`.

---

## ✅ Step 0: Clean Up (Optional but Recommended)

Before creating the new environment:

```bash
# Clear pip cache
pip cache purge

# Clean all conda caches (tarballs, index, logs)
conda clean --all --yes

# Remove old environment if it exists
conda deactivate
conda env remove -n reinforcementLearningProject_2
```

---

## 🌱 Step 1: Create a New Conda Environment

```bash
conda create -n pong-dqn-env python=3.10 -y
conda activate pong-dqn-env
```

- We use **Python 3.10** for compatibility with TensorFlow 2.15 and `gymnasium`.

---

## 📦 Step 2: Install Core Dependencies (Base Packages First)

Install packages in the correct order to avoid version conflicts.

### ➤ 1. Install low-level dependencies first:

```bash
pip install numpy==1.26.4
pip install scipy==1.13.0
pip install pillow==10.3.0
pip install imageio==2.34.0
pip install protobuf==4.25.3
pip install opencv-python==4.9.0.80
```

---

### ➤ 2. Install TensorFlow (core ML engine)

```bash
pip install tensorflow==2.15.0
```

> This includes `keras`, `tensorboard`, and other dependencies.  
> ❗ Do NOT install `tensorflow-gpu` or `tensorflow-estimator` separately.

---

### ➤ 3. Install visualization tool

```bash
pip install matplotlib==3.8.4
```

---

### ➤ 4. Install RL environment dependencies

```bash
pip install gymnasium==0.29.1
pip install ale-py==0.8.1
```

> ✅ `gymnasium` is the maintained version of OpenAI Gym  
> ❌ Do NOT use `gym` or `atari-py` (deprecated)

---

## 🧪 Step 3: (Optional) Extras for Development

```bash
# Only if you want to run notebooks
pip install jupyterlab
```

---

## 📁 Optional: Save Your Environment

To save the environment for future replication:

```bash
pip freeze > requirements.txt
```

Or create a `.yml` version:

```bash
conda env export > pong-dqn-env.yml
```

---

## 🏁 Done!

You can now run the Pong DQN training script confidently without version conflicts or deprecated libraries.



```markdown
# ADT-DDPM (Adaptive Dilated Time-aware DDPM)

This repository contains the implementation of **ADT-DDPM** (Adaptive Dilated Time-aware Denoising Diffusion Probabilistic Models). The code demonstrates the model's capabilities using the **PU (Purdue) Dataset**.

## 📂 Project Structure

### Core Components
- **`Train.py`**
  The main training script. It handles the optimization loop and specifically implements the calculation of the **Intra-batch Diversity Loss** to improve generation diversity.
- **`sample.py`**
  The sampling script used to generate new data samples from the trained model.
- **`test.py`**
  The validation script used to evaluate model performance.

### Model Architectures
- **`model.py`**
  The main **UNet** backbone for the diffusion model. It features:
  - **Adaptive Dilated Convolution**: For dynamic receptive field adjustment.
  - **Adaptive Timestep-aware Attention**: For improved temporal guidance during denoising.
- **`Resnet.py`**
  Implementation of **ResNet18**.
- **`LSTM.py`**
  Implementation of **LSTM**.

### Configuration
- **`DFConfig.py`**
  Configuration file for the **Diffusion Model** parameters (e.g., noise schedules, timesteps).
- **`config.py`**
  Global configuration file containing model hyperparameters and general settings.

### Data
- **`data/`**
  This directory contains the **generated sample data** produced by the model.

## 🚀 Usage

### 1. Data Preparation
The project is configured to work with the **PU Dataset**. Ensure the dataset paths in `config.py` are correctly set.

### 2. Training
To train the ADT-DDPM model with Intra-batch Diversity Loss:
```bash
python Train.py
```

### 3. Sampling
To generate new samples using the trained model:
```bash
python sample.py
```

### 4. Validation
To validate the model:
```bash
python test.py
```

## ⚙️ Requirements
*   Python 3.x
*   PyTorch
*   NumPy
```

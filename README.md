# ADT-DDPM: Adaptive Dilated Time-aware Denoising Diffusion Probabilistic Models

This repository contains the implementation of **ADT-DDPM** (Adaptive Dilated Time-aware Denoising Diffusion Probabilistic Models). The code demonstrates the model's performance and capabilities using the **PU (Purdue) Dataset**.

## 📖 Overview

ADT-DDPM enhances the standard diffusion model architecture by introducing adaptive mechanisms to better capture temporal dependencies and feature details. This implementation focuses on generating high-quality time-series data while maintaining diversity.

## 📂 Project Structure

### Core Logic
*   **`Train.py`**: The main training script. It handles data loading, model optimization, and specifically includes the calculation of the **Intra-batch Diversity Loss** to prevent mode collapse and ensure varied generation.
*   **`sample.py`**: The sampling script used to generate new data samples from the trained model.
*   **`test.py`**: The validation script used to evaluate the model's performance.

### Model Architecture
*   **`model.py`**: Defines the backbone **UNet** architecture. It incorporates two key innovations:
    *   **Adaptive Dilated Convolution**: Dynamically adjusts the receptive field based on input features.
    *   **Adaptive Timestep-aware Attention**: A mechanism to effectively integrate timestep information into the attention layers.
*   **`Resnet.py`**: Implementation of the **ResNet18** architecture (used as a baseline or evaluator).
*   **`LSTM.py`**: Implementation of the **LSTM** architecture (used as a baseline or evaluator).

### Configuration
*   **`DFConfig.py`**: Configuration file specifically for **Diffusion Model** parameters (e.g., noise schedules, total timesteps, sampling settings).
*   **`config.py`**: Global configuration file containing general hyperparameters (e.g., batch size, learning rate) and model dimension settings.

### Data
*   **`data/`**: This directory contains the **generated sample data** produced by the model.

## 🚀 Usage

### 1. Prerequisites
Ensure you have the necessary Python libraries installed (PyTorch, NumPy, etc.).

### 2. Data Preparation
The code is currently configured to utilize the **PU Dataset**. Ensure the dataset paths in `config.py` are correctly pointing to your local data directory.

### 3. Training
To start training the ADT-DDPM model with the specific Intra-batch Diversity Loss:

```bash
python Train.py

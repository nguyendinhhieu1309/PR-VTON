# PR-VTON: Robust 3D Virtual Try-On under Complex Poses

**Nguyen Dinh Hieu, Pham Thi Van Anh, Do Ngoc Bich, Tran Tien Long, Pham Hong Phong**  
FPT University, Hanoi, Vietnam  
[hieundhe180318@fpt.edu.vn](mailto:hieundhe180318@fpt.edu.vn), [vanhphamjuly@gmail.com](mailto:vanhphamjuly@gmail.com), [tinhthanh719@gmail.com](mailto:tinhthanh719@gmail.com), [longtthe176743@fpt.edu.vn](mailto:longtthe176743@fpt.edu.vn), [phongphhe176151@fpt.edu.vn](mailto:phongphhe176151@fpt.edu.vn)

---
This repository cpresents Pose-Robust 3D Virtual Try-On Network (PR-VTON), a novel framework addressing the persistent challenges of achieving realistic 3D virtual try-on under complex human poses. In contrast to prior works that primarily handle frontal or simplified postures, PR-VTON is designed to accommodate extreme variations such as side-facing stances, crossed arms, and severe self-occlusions, conditions that often lead to geometric distortions and inconsistent garment rendering. The proposed approach integrates a personalized diffusion model with a pose-aware 3D Gaussian Splatting editing pipeline, enabling fine-grained garment transfer while preserving high-fidelity geometry and texture across multiple viewpoints. To support training and evaluation, a curated and pre-processed dataset named PR-VTON3D is introduced, containing diverse clothing types and challenging poses that offer realistic scenarios for robust model development. Through a reference-driven multi-view editing strategy and a multi-level attention fusion mechanism, PR-VTON achieves superior cross-view consistency, garment similarity, and visual realism compared to state-of-the-art baselines. Experimental results and user studies demonstrate that the proposed framework significantly enhances the reliability of 3D virtual try-on systems in real-world conditions, establishing a new benchmark for pose-invariant garment transfer.

---
## Pipeline
Proposed 3D Virtual Try-On Framework. The pipeline consists of five stages: (1) the Monocular Prediction Module (MPM) performs garment alignment, segmen-tation, and double-depth estimation; (2) the Depth Refinement Module (DRM) recovers fine-grained geometry; (3) reference-driven image editing enforces cross-view consistency; (4) a personalized diffusion model with LoRA fine-tuning synthesizes garment textures under occlusions; and (5) persona-aware 3D Gaussian Splatting (3DGS) editing and rendering integrate geometry, texture, and identity features into high-fidelity multi-view try-on results.

<img width="588" height="584" alt="image" src="https://github.com/user-attachments/assets/5e3cecd0-7308-4a7f-b5fc-f5bbb58022d4" />

---

## ⚙️ Framework and Environment Setup  

This project utilizes the following core frameworks and libraries:  

| Framework | Version |  
|-----------|---------|  
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1%2Bcu118-ee4c2c?logo=pytorch&logoColor=white) | 2.2.1 + cu118 |  
| ![TorchVision](https://img.shields.io/badge/TorchVision-0.17.1%2Bcu118-3776ab?logo=pytorch-lightning&logoColor=white) | 0.17.1 + cu118 |  
| ![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white) | 3.12 |  
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-5C3EE8?logo=opencv&logoColor=white) | 4.10.0 |  
| ![CuPy](https://img.shields.io/badge/CuPy-13.3.0-00a95c?logo=numpy&logoColor=white) | 13.3.0 |  
| ![TensorBoard](https://img.shields.io/badge/TensorBoard-2.4-FF6F00?logo=tensorflow&logoColor=white) | 2.4 |  
| ![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.x-792ee5?logo=pytorchlightning&logoColor=white) | latest |  
| ![Diffusers](https://img.shields.io/badge/Diffusers-0.25.0-ffca28?logo=huggingface&logoColor=white) | 0.25.0 |

---

## 🚀 Installation

```bash
# Clone repo
git clone https://github.com/nguyendinhhieu1309/PR-VTON.git
cd pr-vton

# Create environment
conda create -n prvton python=3.12 -y
conda activate prvton

# Install PyTorch + CUDA
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

---

## PR-VTON3D Dataset  

We release **PR-VTON3D**, a high-resolution, multi-view dataset containing **156 subjects** with diverse poses and occlusion scenarios.  

- **12–16 synchronized views per subject**  
- **38,400 RGB images at 2048×2048**  
- Paired textured meshes reconstructed via photogrammetry  

👉 Please check [DATA_PREP.md](./DATA_PREP.md) for details on dataset access and preprocessing.  

### Mini-Data Download  
For quick testing, you can download a **mini version** of PR-VTON3D here:  
[Download MiniData Train](https://drive.google.com/drive/folders/1wsIp7n2msLdNLffNo4EEKPfWZZK_284w?usp=drive_link)
[Download MiniData Test](https://drive.google.com/drive/folders/13btss4VdyG6R7R9mLTmzqqssg_7YCaIf?usp=drive_link)

Unzip and place under:  

```
DATA/
├── mini_data/
    ├── images/
    ├── point_cloud/
    ├── transforms.json
```

---

## Pre-trained Model Preparation  

Download the following pre-trained weights before running PR-VTON:  

- [DensePose](https://example.com/densepose-weights)  
- [OpenPose](https://example.com/openpose-weights)  
- [Human Parsing](https://example.com/humanparsing-weights)  
- [Self-Correction Parsing](https://example.com/self-correction-weights)  
- [Double-Depth Estimation](https://example.com/depth-weights)  
- [Stable Diffusion v1.5](https://example.com/stable-diffusion-v1-5)  
- [LoRA Garment Weights](https://example.com/lora-garment)  

We provide pre-trained models for quick start.  
👉 [Download Pre-trained Models](https://drive.google.com/drive/folders/14HpZlA9KLJtvb8pSsIbXjNL93BZ_RO6b?usp=drive_link)

Expected folder structure:

```
stage1/
├── ckpt/
│   ├── densepose/
│   │   ├── model_final_162be9.pkl
│   ├── openpose/
│   │   ├── ckpts/
│   │       ├── body_pose_model.pth
│   ├── humanparsing/
│   │   ├── parsing_lip.onnx
│   │   ├── parsing_atr.onnx

stage2/
├── Self_Correction_Human_Parsing/
│   ├── logits.pt
├── depth_estimation/
│   ├── depth_front.pth
│   ├── depth_back.pth

diffusion/
├── stable-diffusion-v1-5/
│   ├── model.ckpt
├── lora/
│   ├── garment_lora.pth
```

---

## Data Preparation  

1. Follow [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) to process your own video data.  
2. Use [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) to initialize the 3DGS point cloud.  

After preparation, your data folder should look like:  

```
DATA/
├── stereo/
├── input/
├── sparse/
├── point_cloud/
│   ├── iteration_30000/
│       ├── point_cloud.ply
├── images/
├── distorted/
├── sparse_pc.ply
├── input.ply
├── transforms.json
├── cameras.json
```

---

## Running PR-VTON  

```bash
python3 main.py   --data_path {/PATH/TO/PROCESSED_DATA}   --gs_source {/PATH/TO/PROCESSED_DATA/point_cloud/iteration_30000/point_cloud.ply}   --cloth_path {/PATH/TO/GARMENT/IMAGE}
```

---

## Citation  

If you use this repository, please cite:  

```bibtex
@article{nguyen2025prvton,
  title={PR-VTON: Pose-Robust 3D Virtual Try-On with Gaussian Splatting},
  author={Nguyen, Dinh Hieu and Pham, Thi Van Anh and Do, Ngoc Bich and Tran, Tien Long and Pham, Hong Phong},
  journal={Proc. MIWAI},
  year={2025}
}
```

# SparseEnd2End: Obstacle 3D Detection and Tracking Architecture Based VisionTransformer
- 👋 Hi, I’m ThomasVonWu. I'd like to introduce you to a  simple and practical repository that uses end-to-end model with sparse transformer to sense 3D obstacles. This repo has no complex dependency for Training | Inference | Deployment(we don't need to install mmdet3d, mmcv, mmcv-full and so on), so it's easy to install in your local workstation or supercomputing gpu clusters. This repository will also provide x86(NVIDIA RTX  Series GPU) | ARM(NVIDIA ORIN) deployment solutions. Finally, you can deploy your e2e model onborad through this repo happily.  
- 👀 I think you are interested in:  
    1. how to define and use PyTorch custom operations : DeformableAggregation?    
    2. how to export a PyTorch model with custom operations to ONNX?  
    3. how to export onnx with custom operations to TensorRT engine?  
    4. how to validate PyTorch and TensorRT inference results consistency?  

## News
* **`25 Aug, 2024`:** I release repo: SparseEnd2End. The complete deployment solution will be released as soon as possible. Please stay tuned! 

## Tasklist
- [X] *Register custom operation : DeformableFeatureAggregation and export ONNX and TensorRT engine. **`25 Aug, 2024`***
- [X] *Verify the consistency of reasoning results : DeformableFeatureAggregation  PyToch Implementation  vs. TensorRT plugin Implementation. **`25 Aug, 2024`***
- [] *Reasoning acceleration using CUDA shared memory and CUDA FP16 in DeformableFeatureAggregation plugin Implementation.*
- [ ] *Export SparseTransFormer Backbone ONNX&TensorRT engine.*
- [ ] *Verify the consistency of reasoning results : SparseTransFormer Backbone PyTorch Implementation vs. ONNX Runtime vs. TensorRT engine.*
- [ ] *Export SparseTransFormer Head ONNX and TensorRT engine.*
- [ ] *Verify the consistency of reasoning results : SparseTransFormer Head PyTorch Implementation vs. TensorRT engine.*
- [ ] *Reasoning acceleration using FlashAttention in replace of MultiheadAttention.*
- [ ] *Reasoning acceleration using FP16/INT8  in replace of FP32 of TensorRT engine.*
- [] *Image pre-processing Instancbank Caching and model post-processing Implementation with C++.*
- [] *Reasoning acceleration : Image pre-processing Instancbank Caching and model post-processing Implementation with CUDA.*
- [ ] *Full-link reasoning using CUDA, TensorRT and C++.*

# Introduction
> SparseEnd2End is a Sparse-Centric paradigm for end-to-end autonomous driving perception.  

## Quick Start
[Quick Start](QUICK-START.md)

## Citation
If you find SparseEnd2End useful in your research or applications, please consider giving us a star &#127775;  

## 🏷 ChangeLog
>**08/25/2024：** **[v1.0.0]** This repo now supports Training | Inference in NuscenesDataset. It includes: data dump in JSON, Training | Inference  log caching, TensorBoard hooking, and so on. 
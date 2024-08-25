# SparseEnd2End: Obstacle 3D Detection and Tracking Architecture Based VisionTransformer
- 👋 Hi, I’m ThomasVonWu. I'd like to introduce you to a practical and simple repository that uses sparse transformer end-to-end models to sense 3D obstacles. This repo has no complex dependency for training&inference(we don't need install mmdet3d, mmcv, mmcv-full and so on), so it's easy to use in local workstation or supercomputing gpu clusters. This repository will alse provide x86 and orin-based deployment solutions. Finally, you can deploy your e2e model onborad through this repo happily.  
- 👀 I think you are interested in this knowledge below:  
    1. how to define and use PyTorch custom operations : DeformableAggregation?    
    2. how to export a PyTorch model with custom operations to ONNX?  
    3. how to export onnx with custom operations to TensorRT engine?  
    4. how to validate PyTorch and TensorRT inference results consistency?  

## News
* **`25 Aug, 2024`:** We release code SparseEnd2End repo. Deployment Code will be released as soon as possible. Please stay tuned! 

## Tasklist
- [x] DeformableFeatureAggregation  PyTorch&ONNX&TensorRT custom operations has been registered and exports ONNX&TensorRT engine successfully. **`25 Aug, 2024`**  
- [x] DeformableFeatureAggregation  plugin has completed consistency verification of inference results. **`25 Aug, 2024`:**  
- [ ] Export SparseTransFormer Backbone ONNX&TensorRT engine.
- [ ] Consistency verification of inference results between PyTorch model & ONNX & TensorRT engine of SparseTransFormer Backbone .
- [ ] Export SparseTransFormer Head ONNX&&TensorRT engine.
- [ ] Consistency verification of inference results between PyTorch model & TensorRT engine of SparseTransFormer Head.

# Introduction
> SparseEnd2End is a Sparse-Centric paradigm for end-to-end autonomous driving perception.  
-
-
-

## Quick Start
[Quick Start](QUICK-START.md)

## Citation
If you find SparseEnd2End useful in your research or applications, please consider giving us a star &#127775;  

## 🏷 ChangeLog
>**08/25/2024：** **[v1.0.0]** Our repo now supports training&inference in NuscenesDataset. This includes: data dump in JSON, training&inference log caching, metric cache locally, TensorBoard hookS, and so on. 
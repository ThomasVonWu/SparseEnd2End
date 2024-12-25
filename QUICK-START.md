## 📫 Quick Start  SparseEnd2End LocalWorkStation | GPU Cluster

### Git clone Repository
```bash
git clone --recursive https://github.com/ThomasVonWu/SparseEnd2End.git
cd SparseEnd2End
```
If you forget to add `--recursive` in the Git command line or want to update submodules in SparseEnd2End, run the following command:
```bash
git clone https://github.com/ThomasVonWu/SparseEnd2End.git
cd SparseEnd2End
git submodule init
git submodule update --recursive --remote
```

### Set up a new virtual environment and install packpages using pip3
```bash
virtualenv sparsee2e --python=python3.8
source path/to/sparsee2e/bin/activate
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install colored polygraphy --extra-index-url pypi.ngc.nvidia.com
pip3 install -r requirement.txt
```

### Use a Docker image instead of the above method: setting up a virtual environment locally.
```bash
# STEP-1 Build docker image:
# Selection1 : NVIDIA A100/V100/H20
cd docker/cuda121_cudnn8_ub2004_torch210_py310_torchvision016_torchaudio_210
docker build -t sparsee2e:v1.0 .
cd -

or 

# Selection2 : NVIDIA RTX3090
cd docker/cuda1116_cudnn8_ub2004_torch1300_py38_torchvision014_torchaudio_0130
docker build -t sparsee2e:v1.0 .
cd -

# STEP-2 Create docker container (you need install nvidia-docker)
sudo nvidia-docker run \
    -it \
    -d \
    --shm-size 64g \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/SparseEnd2End:/path/to/SparseEnd2End \
    -v /path/to/nuscenes_datas:/path/to/nuscenes_datas \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY="${DISPLAY}" \
    -w /path/to/SparseEnd2End  \
    --name sparsee2e_container \
    --user root \
    sparsee2e:v1.0 \
    bash

# STEP-3 Run your docker container
docker exec -it sparsee2e_container bash
```

### Compile the deformable_feature_aggregation and  focal_loss CUDA op
```bash
cd modules/ops
python3 setup.py develop
cd ../..
cd modules/head/loss/base_loss/ops
python3 setup.py develop
cd ../../../../../
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download) and create symbolic links.
```bash
cd ${sparseend2end_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required .json files.
```bash
export json_dir="data/nusc_anno_dumpjson"
export nusc_ver=v1.0-mini # or  v1.0-trainval
mkdir -p ${json_dir}
python3 script/dump_nusc_ann_json.py --version ${nusc_ver}
```

### Generate anchors by K-means
```bash
# Specify file path:
python3 script/nuc_anchor_generator.py --ann_file data/nusc_anno_dumpjson/train/[TOKENID].json # e.g. TOKENID = fcbccedd61424f1b85dcbf8f897f9754
or
# Specify folder path:
python3 script/nusc_anchor_generator.py --ann_file_dir data/nusc_anno_dumpjson/train
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Train&Val Pipeline Data Visualization
train_pipeline or val_pipeline online/offline separately for visualization.
```bash
python script/tutorial_task_nusc/001_nusc_dataset_visualization.py
```


### Train&Test Pipeline in DP Mode
Set PythonEnvironment
```bash
export PYTHONPATH=$PYTHONPATH:./
```

Train in dp mode.
```bash
export  config_path=dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py 
# Method1
clear && python script/train.py ${config_path} 

# Method2
# Using "no-validate" to avoid getting the invalid box type error during evaluation after each epoch.
clear && python script/train.py ${config_path}  --no-validate
```

Test in dp mode
```bash
export  config_path=dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py 
clear && python script/test.py  ${config_path} --checkpoint /path/to/checkpoint
```

### Train in DDP Mode.
```bash
export  config_path=dataset/config/sparse4d_temporal_r50_1x4_bs22_256x704.py  # defaule training in nuscenes v1.0-trainval with 4GPUs.
export python_version=python3.10 # your environment python version(or 3.8)
sed -i 's/self.class_range.keys()/list(self.class_range.keys())/g' /opt/conda/lib/${python_version}/site-packages/nuscenes/eval/detection/data_classes.py

# Method1
export gpu_nums=4
export port=28650
python3 -m torch.distributed.launch --nproc_per_node=${gpu_nums} --master_port=${port} script/train.py  --no-validate  --launcher pytorch

# Method2
clear && bash script/dist_train.sh ${config_path}
```

###  Evaluation Offline
```bash
# If you use  `--no-validate` args while launching train script, you can evaluation checkpoints offline.
export iter_per_epoch=79 # if your config is the same with `sparse4d_temporal_r50_1x4_bs22_256x704.py`,  iter_per_epoch=int(28130 // (4 * 88))
python3 script/offline_eval.py ---cfg /path/to/train/cfg -dir_path /path/to/train/cached/ckpt/dir/path --iter_per_epoch ${iter_per_epoch}
```
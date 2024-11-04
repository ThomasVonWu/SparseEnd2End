
# 📫 Quick Start LocalWorkStation | GPU Cluster
### Set up a new virtual environment
```bash
virtualenv sparsee2e --python=python3.8
source path/to/sparsee2e/bin/activate
```

### Install packpages using pip3
```bash
sparseend2end_path="path/to/sparseend2end"
cd ${sparseend2end_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
python3 -m pip install colored polygraphy --extra-index-url pypi.ngc.nvidia.com
pip3 install -r requirement.txt
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
json_dir="data/nusc_anno_dumpjson"
nusc_ver=v1.0-mini # or  v1.0-trainval
mkdir -p ${json_dir}
python3 script/dump_nusc_ann_json.py --version ${nusc_ver}
```

### Generate anchors by K-means
```bash
# Specify file path:
python3 script/nuc_anchor_generator.py --ann_file data/nusc_anno_dumpjson/train/[tokenid].json
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


### Train&Val Pipeline in DP Mode
Set PythonEnvironment
```bash
export PYTHONPATH=$PYTHONPATH:./
```

Train in dp mode.
```bash
config_path=dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py # or dataset/config/sparse4d_temporal_r50_1x4_bs22_256x704.py
clear && python script/train.py ${config_path} 
or 
# Using "no-validate" to avoid getting the invalid box type error during evaluation after each epoch.
clear && python script/train.py ${config_path}  --no-validate
```

Test in dp mode
```bash
clear && python script/test.py  ${config_path} --checkpoint /path/to/checkpoint
```


Train in ddp mode.
```bash
python_version=python3.10
sed -i 's/self.class_range.keys()/list(self.class_range.keys())/g' /opt/conda/lib/${python_version}/site-packages/nuscenes/eval/detection/data_classes.py

port=28650
gpu_nums=4
python3 -m torch.distributed.launch --nproc_per_node=${gpu_nums} --master_port=${port} script/train.py  --no-validate  --launcher pytorch
or 
clear && bash script/dist_train.sh ${config_path}
```

Evaluation Offline
```bash
# If you use -- ,you can evaluation offline
python script/offline_eval.py ---cfg /path/to/train/cfg -dir_path /path/to/train/cached/ckpt/dir/path --iter_per_epoch [int] 
```

# 📫 Quick Start LocalWorkStation
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
mkdir -p ${json_dir}
python3 script/dump_nusc_ann_json.py --version [v1.0-mini/v1.0-trainval]
```

### Generate anchors by K-means
```bash
json_path="data/nusc_anno_dumpjson/train/*.json"
python3 script/nuc_anchor_generator.py --ann_file ${json_path}
or
python3 script/nuc_anchor_generator.py --ann_file_dir $json_dir"/train"
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

Train&Test in dp mode.
```bash
clear && python script/train.py dataset/config/[your_config.py]
clear && python script/test.py  dataset/config/[your_config.py] --checkpoint path/to/checkpoint
```

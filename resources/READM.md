# Add/Update sparse_end2end GIF 

**1st Step**
```python
cd /path/to/SparseEnd2End
python script/tutorial/001.nusc_dataset_visualization.py
```
**2nd Step**
```bash
sudo apt install ffmpeg

# Method1 low-quality
ffmpeg -i sparse_end2end.mp4 -vf "fps=10,scale=1312:-1:flags=lanczos" resources/sparse_end2end.gif

# Method2 high-quality
ffmpeg -i sparse_end2end.mp4 -vf fps=10,scale=1312:-1:flags=lanczos,palettegen resources/sparse_end2end.png                           

ffmpeg -i sparse_end2end.mp4 -i resources/sparse_end2end.png -filter_complex "fps=10,scale=1312:-1:flags=lanczos[x];[x][1:v]paletteuse" resources/sparse_end2end.gif
```
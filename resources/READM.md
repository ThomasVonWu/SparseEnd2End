# Add/Update sparse_end2end GIF 

```bash
cd /path/to/SparseEnd2End
sudo apt install ffmpeg
ffmpeg -i /path/to/sparse_end2end.mp4 -vf "fps=10,scale=1312:-1:flags=lanczos" resources/sparse_end2end.gif
```
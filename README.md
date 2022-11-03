# Unified Implicit Neural Stylization
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[[Paper]](https://arxiv.org/abs/2204.01943) [[Website]](https://zhiwenfan.github.io/INS/)

<div>
<img src="https://raw.githubusercontent.com/zhiwenfan/INS/main/INS/ours_lego_inter.gif" height="140"/>
<img src="https://raw.githubusercontent.com/zhiwenfan/INS/main/INS/ours_lego_the_scream.gif" height="140"/>
<img src="https://raw.githubusercontent.com/zhiwenfan/INS/main/INS/ours_mic_starrynight.gif" height="140"/>
<img src="https://raw.githubusercontent.com/zhiwenfan/INS/main/INS/ours_room.gif" height="140"/>
<img src="https://raw.githubusercontent.com/zhiwenfan/INS/main/INS/ours_horns_gris1.gif" height="140"/>
</div>

## Installation

We recommend users to use `conda` to install the running environment. The following dependencies are required:
```
pytorch=1.7.0
torchvision=0.8.0
cudatoolkit=11.0
tensorboard=2.7.0
opencv
imageio
imageio-ffmpeg
configargparse
scipy
matplotlib
tqdm
mrc
lpips
```

## Data Preparation

To run our code on NeRF dataset, users need first download data from official [cloud drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then extract package files according to the following directory structure:

```
├── configs
│   ├── ...
│
├── datasets
│   ├── nerf_llff_data
│   │   └── room
│   │   └── horns   # downloaded llff dataset
|   |   └── ...
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── chair    # downloaded synthetic dataset
|   |   └── ...
```
The last step is to generate and process data via our provided script:
```
python gen_dataset.py --config <config_file>
```
where `<config_file>` is the path to the configuration file of your experiment instance. Examples and pre-defined configuration files are provided in `configs` folder.

### Download Prepared Data:
We provide a data sample for scene "room" in the [Google_Drive](https://drive.google.com/file/d/1W0jtIAu0el04awnSNp712gHZTQJ9nGiU/view?usp=sharing), you can direct download it without any modification.


## Testing

After generating datasets, users can test the conditional style interpolation of INS+NeRF by the following command:
```
bash scripts/linear_eval.sh
```
Inference on scene-horns with style-gris1:
```
bash scripts/infer_horns.sh
```

## Training

One can do training using:
```
bash scripts/train_room_thescream_28G_mem.sh
```

## Stylizing Textured SDF

We also provide code and scripts to stylize textured signed distance functions based on [Implicit Differentiable Renderer (IDR)](https://arxiv.org/abs/2003.09852).

To prepare data, run scripts `data/download_data.sh`, which will download the DTU dataset into the `datasets/` directory.

Afterwards, train an IDR for a scanned data in DTU where the available IDs are listed in `datasets/DTU`:
```
python run_idr.py --conf ./configs/idr_fixed_cameras.conf --scan_id <SCAN_ID>
```

Finally, one can stylize an IDR model with a style image specified in the configuration file:
```
python run_idr.py --conf <CONF_FILE> --scan_id <SCAN_ID> --is_continue
```
in which we defined two preset configurations `configs/idr_stylize_face.conf` and `configs/idr_stylize_scream.conf`.

## Citation

If you find this repo is helpful, please cite:

```
@inproceedings{fan2022unified,
  title={Unified Implicit Neural Stylization},
  author={Fan, Zhiwen and Jiang, Yifan and Wang, Peihao and Gong, Xinyu and Xu, Dejia and Wang, Zhangyang},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

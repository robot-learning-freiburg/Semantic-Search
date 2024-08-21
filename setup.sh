#!/usr/bin/bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
conda env create -n habitat --file environment.yaml
source ~/.bashrc
conda activate habitat
# need username password/token for the hm3d dataset
python -m habitat_sim.utils.datasets_download --uids hm3d --data-path data/ --username $USERNAME --password $PASSWORD

ln -s $(pwd)/data/versioned_data/hm3d-0.2/hm3d $(pwd)/data/scene_datasets/hm3d_v0.2

wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
mkdir -p data/datasets/objectnav/hm3d/
unzip objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d
mv data/datasets/objectnav/hm3d/objectnav_hm3d_v2 data/datasets/objectnav/hm3d/v2
rm objectnav_hm3d_v2.zip

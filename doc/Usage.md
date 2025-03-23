#### step 0. Installation. Please use the following script to set up the Python environment
```
git clone https://github.com/Runsong123/Unified-Lift.git
cd Unified-Lift

conda create -n unifed_lift python=3.8 -y
conda activate unifed_lift 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
#### step 1. Preparing the data. You can download the LERF-Masked dataset from this [link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask).


#### step 2. Training Unified-Lift (Gaussian-level features + object-level codebook). You can use the training scripts to train on the LERF-Masked dataset.
```
bash train.sh
```

#### step 3. Once you have finished training, you can directly predict segmentation. Use the following script to export the evaluation results.
```
bash infer.sh
```
# !/bin/bash
SBATCH --nodes=1
SBATCH --ntasks-per-node=1
SBATCH --cpus-per-task=1
SBATCH --time=24:00:00
SBATCH --mem=16GB
SBATCH --job-name=rd2
SBATCH --output=/home/tf2387/7123pj_zzp/outputs/rd2.out
SBATCH --gres=gpu:1
SBATCH --account=pr_31_tandon_advanced

# Create necessary directories
mkdir -p /home/tf2387/7123pj_zzp/outputs

# Clean modules and prepare environment
module purge

# Run with Singularity container
singularity exec --nv --overlay /scratch/yh3986/singularity_utils/pcrender-overlay-v100-50G-10M.ext3:ro\
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/env.sh && conda activate /ext3/envs/pcrender && \
    
    cd /home/tf2387/7123pj_zzp
    
    python train.py \
      --data-dir dataset/cifar-10-python/cifar-10-batches-py \
      --test-data dataset/cifar-10-python/cifar-10-batches-py/test_batch \
      --task train \
      --output-dir outputs/rd2 \
      --model-size medium \
      --batch-size 128 \
      --epochs 200 \
      --lr 0.1 \
      --optimizer sgd \
      --scheduler cosine \
      --weight-decay 5e-4 \
      --activation relu \
      --dropout
  "

python train.py       --data-dir dataset/cifar-10-python/cifar-10-batches-py       --test-data dataset/cifar-10-python/cifar-10-batches-py/test_batch       --task test      --output-dir /scratch/rg4827/hw/dl1/outputs/no-mix-800_2025-03-13_16-08-22       --model-size medium       --batch-size 256   --epochs 600       --lr 0.1       --optimizer sgd       --scheduler cosine       --weight-decay 5e-4       --activation relu   --dropout


python train.py       --data-dir dataset/cifar-10-python/cifar-10-batches-py       --test-data dataset/cifar-10-python/cifar-10-batches-py/test_batch       --task train      --output-dir outputs/no-mix-800       --model-size medium       --batch-size 256   --epochs 800       --lr 0.1       --optimizer sgd       --scheduler cosine       --weight-decay 5e-4       --activation relu   --dropout

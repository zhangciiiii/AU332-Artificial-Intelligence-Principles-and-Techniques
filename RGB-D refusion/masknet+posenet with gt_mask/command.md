```
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset_dir /home/stu_4/refusion/pre-png  --batch-size 32    \
--mask-loss-weight 0.05  --smooth-loss-weight 0.05 --consensus-loss-weight 2.0  --pose-loss-weight 2.0 \
--epoch-size 1000  --nlevels 6 --lr 1e-4 -wssim 0.7   \
--epochs 200  --name gt_mask
```

train posenet with reconstruction images;

train masknet with the gt mask from semantic segmentation.

python train_sam2_hair.py \
    --images_csv hair_sam_preparation/audit/images.csv \
    --instances_csv hair_sam_preparation/audit/instances.csv \
    --max_train_instances 2000 \
    --max_val_instances 500 \
    --epochs 1 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --workers 8


python train_sam2_hair.py \
    --images_csv hair_sam_preparation/audit/images.csv \
    --instances_csv hair_sam_preparation/audit/instances.csv \
    --output_dir sam2_hair_runs/base_plus_stage1 \
    --max_train_instances 100000 \
    --max_val_instances 10000 \
    --epochs 3 \
    --batch_size 1 \
    --accumulation_steps 4 \
    --workers 8 \
    --lr 1e-5 \
    --box_jitter 20
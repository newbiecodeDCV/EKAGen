#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
  --anno_path ./dataset/iu_xray/annotation_png_fixed.json \
  --data_dir ./dataset/iu_xray \
  --dataset_name iu_xray \
  --image_size 300 \
  --batch_size 16 \
  --epochs 50 \
  --dec_layers 6 \
  --enc_layers 6 \
  --dim_feedforward 2048 \
  --hidden_dim 256 \
  --num_heads 8 \
  --dropout 0.1 \
  --lr 1e-4 \
  --lr_backbone 1e-5 \
  --backbone resnet101 \
  --max_position_embeddings 128 \
  --num_classes 14 \
  --t_model_weight_path ./weight_path/mimic_t_model.pth \
  --detector_weight_path ./weight_path/diagnosisbot.pth \
  --weight_path ./weight_path/iu_weight.pth \
  --knowledge_prompt_path ./knowledge_path/knowledge_prompt_iu.pkl \
  --thresholds_path ./thresholds.pkl \
  --seed 42 \
  --output_dir ./outputs/iu_xray_test

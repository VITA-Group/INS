#! /in/bash

SCENE=room
BATCH_SIZE=1
# sentitive to gpu memory, lower it if gpu mem is small
PATCH_SIZE=72
# related to receptive field
PATCH_STRIDE=2
RGB_W=0
STYLE_W=1e11
CONTENT_W=1
DENSITY_W=1e6
EXPNAME=${SCENE}_thescream_P${PATCH_SIZE}_PS${PATCH_STRIDE}_RGB${RGB_W}_S${STYLE_W}_C${CONTENT_W}_D${DENSITY_W}_updateAll
mkdir -p logs/$EXPNAME
mkdir -p logs/$EXPNAME/checkpoints

CUDA_VISIBLE_DEVICES=1 python run_nerf.py \
 --batch_size ${BATCH_SIZE} \
 --config configs/${SCENE}.txt  \
 --data_path datasets/nerf_llff_data/${SCENE}   \
 --expname $EXPNAME    \
 --with_teach  \
 --d_weight $DENSITY_W  \
 --content_weight $CONTENT_W  \
 --rgb_weight ${RGB_W}   \
 --style_weight  ${STYLE_W}  \
 --patch_size ${PATCH_SIZE}  \
 --N_iters 300000  \
 --loss_terms coarse fine style_v_all density    \
 --fix_param False False \
 --style_path datasets/single_styles/the_scream.jpg \
 --i_testset 2500  \
 --i_video 10000  \
 --i_weights 2500 \
 --patch_stride ${PATCH_STRIDE} \
 --stl_idx 0 \
 --ckpt_path ckpts/${SCENE}_00170000.ckpt 2>&1 | tee -a logs/${EXPNAME}/${EXPNAME}.txt
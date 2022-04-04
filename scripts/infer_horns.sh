CUDA_VISIBLE_DEVICES=3  python run_nerf.py  \
    --batch_size 1 \
    --config configs/horns.txt  \
    --data_path datasets/nerf_llff_data/horns/   \
    --expname  horns_github    \
    --ckpt_path ckpts/horns_gris1.ckpt    \
    --render_video
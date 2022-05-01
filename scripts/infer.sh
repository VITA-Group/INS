CUDA_VISIBLE_DEVICES=0  python run_nerf.py  \
    --batch_size 1 \
    --config configs/horns.txt  \
    --data_path datasets/nerf_llff_data/horns/   \
    --expname  horns_gris1    \
    --ckpt_path ckpts/horns_gris1.ckpt    \
    --render_video


CUDA_VISIBLE_DEVICES=1  python run_nerf.py  \
    --batch_size 1 \
    --config configs/lego.txt  \
    --data_path datasets/nerf_synthetic/lego/   \
    --expname  lego_thescream    \
    --ckpt_path ckpts/lego_thescream.ckpt    \
    --render_video

CUDA_VISIBLE_DEVICES=1  python run_nerf.py  \
    --batch_size 1 \
    --config configs/room.txt  \
    --data_path datasets/nerf_llff_data/room/   \
    --expname  room_thescream    \
    --ckpt_path ckpts/room_thescream.ckpt    \
    --render_video
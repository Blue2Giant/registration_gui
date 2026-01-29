cd /data/flux_kontext/matching_anything
roma=/mnt/jfs/model_zoo/maching_anything/weights/matchanything_roma.ckpt
eloftr=/mnt/jfs/model_zoo/maching_anything/weights/matchanything_eloftr.ckpt
python /data/flux_kontext/matching_anything/registration_demo.py /data/flux_kontext/matching_anything/config.py  \
  --ckpt_path $roma \
  --method matchanything_roma@-@ransac_affine \
  --img0 /data/flux_kontext/output/Optical-SAR/pair1_1.jpg \
  --img1 /data/flux_kontext/output/Optical-SAR/pair1_2.jpg \
  --output_dir /data/flux_kontext/output/registration \
  --imgresize 832 \
  --plot_matches \
  --save_chessboard \
  --chessboard_tile 128
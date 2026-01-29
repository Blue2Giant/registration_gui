cd /data/flux_kontext/matching_anything
roma=/mnt/jfs/model_zoo/maching_anything/weights/matchanything_roma.ckpt
eloftr=/mnt/jfs/model_zoo/maching_anything/weights/matchanything_eloftr.ckpt
python /data/flux_kontext/matching_anything/srif_registration.py /data/flux_kontext/matching_anything/config.py  \
  --ckpt_path $roma \
  --method matchanything_roma@-@ransac_affine \
  --pairs_dir /data/flux_kontext/output/Optical-SAR \
  --output_dir /data/loftr_pairs_eval_out \
  --plot_matches \
  --save_chessboard \
  --num_samples 2000 \
  --gt_direction 1to2
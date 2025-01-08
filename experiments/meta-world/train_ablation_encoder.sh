# python run_experiments.py --algorithm fusenet --start-mode 0 --tag FuseHeads --fuse_heads
python run_experiments.py --algorithm fusenet_merge \
                        --start-mode 0 --tag Ablation_Encoder_From_Base \
                        --fuse_heads --pool_size 21 --seed 42 --encoder_from_base --debug
# python run_experiments.py --algorithm simple
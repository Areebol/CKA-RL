# python run_experiments.py --algorithm fusenet --start-mode 0 --tag FuseHeads --fuse_heads
python run_experiments.py --algorithm fusenet_merge \
                        --start-mode 0 --tag woMerge \
                        --fuse_heads --pool_size 21 --seed 1
# python run_experiments.py --algorithm simple
# python run_experiments.py --algorithm fusenet --start-mode 0 --tag FuseHeads --fuse_heads
python run_experiments.py --algorithm fusenet_merge \
                        --start-mode 8 --tag woMerge \
                        --fuse_heads --pool_size 21
# python run_experiments.py --algorithm simple
export CUDA_VISIBLE_DEVICES=2
python run_experiments.py --algorithm fusenet --tag FuseShared\
                        --start-mode 0 \
                        --fuse_shared \
                        --fuse_heads \
                        --pool_size 21 \
                        --seed 42
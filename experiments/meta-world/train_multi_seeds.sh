# export CUDA_VISIBLE_DEVICES=0
# python run_experiments.py --algorithm componet \
#                         --start-mode 0 --tag CompoNets \
#                         --seed 0
# export CUDA_VISIBLE_DEVICES=3
# python run_experiments.py --algorithm componet \
#                         --start-mode 0 --tag CompoNets \
#                         --seed 1
# export CUDA_VISIBLE_DEVICES=1
# python run_experiments.py --algorithm componet \
#                         --start-mode 0 --tag CompoNets \
#                         --seed 2
# export CUDA_VISIBLE_DEVICES=6
# python run_experiments.py --algorithm componet \
#                         --start-mode 0 --tag CompoNets \
#                         --seed 3
# export CUDA_VISIBLE_DEVICES=7
python run_experiments.py --algorithm fusenet --tag Baseline \
                        --start-mode 0 \
                        --fuse_heads \
                        --pool_size 21 \
                        --seed 0
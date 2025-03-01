export CUDA_VISIBLE_DEVICES=2
python run_experiments.py --algorithm componet \
                        --start-mode 13 --tag CompoNets \
                        --seed 1
                        # --start-mode 0 \
                        # --seed 42 --debug

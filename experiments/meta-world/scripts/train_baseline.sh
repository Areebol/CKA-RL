export CUDA_VISIBLE_DEVICES=1
python run_experiments.py --algorithm simple \
                        --start-mode 0 --tag Baseline \
                        --seed 42
                        # --start-mode 0 \
                        # --seed 42 --debug

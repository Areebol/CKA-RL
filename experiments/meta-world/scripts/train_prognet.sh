export CUDA_VISIBLE_DEVICES=5
python run_experiments.py --algorithm prognet \
                        --start-mode 0 --tag Baseline \
                        --seed 42
                        # --start-mode 0 \
                        # --seed 42 --debug

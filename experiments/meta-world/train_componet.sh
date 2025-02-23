export CUDA_VISIBLE_DEVICES=2
python run_experiments.py --algorithm componet \
                        --start-mode 0 --tag Baseline \
                        --seed 42
                        # --start-mode 0 \
                        # --seed 42 --debug

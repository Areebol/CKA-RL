export CUDA_VISIBLE_DEVICES=3
python run_experiments.py --algorithm finetune \
                        --start-mode 0 --tag Baseline \
                        --seed 42
                        # --start-mode 0 \
                        # --seed 42 --debug

SEED=42
TRAIN_MODES="--first-mode 2 --last-mode 7"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-3 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-3"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-4 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-4"
python run_experiments.py --method_type FuseNet \
                          --alpha_learning_rate 1e-2 \
                          --alpha_factor 1e-3 \
                          --fix_alpha True 
                          $TRAIN_MODES \
                          --seed $SEED \
                          --tag "Alpha_lr_1e-2"
                        #   --debug True


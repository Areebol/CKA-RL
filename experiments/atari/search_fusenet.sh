SEED=42
TRAIN_MODES="--first-mode 0 --last-mode 7"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-3 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-3"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-4 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-4"
# TAT = theta + alpha tau 
python run_experiments.py --method_type FuseNet \
                          --alpha_learning_rate 2.5e-4 \
                          --delta_theta_mode T \
                          --alpha_factor 1e-3 \
                          --fuse_actor \
                          --tag "Debug" \
                          $TRAIN_MODES \
                          --seed $SEED \
                        #   --debug
                        #   --fuse_encoder \

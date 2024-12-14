SEED=42
TRAIN_MODES="--first-mode 0 --last-mode 7"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-3 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-3"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-4 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-4"
python run_experiments.py --method_type FuseNet  --fix_alpha True $TRAIN_MODES --seed $SEED --tag "FixAlpha"


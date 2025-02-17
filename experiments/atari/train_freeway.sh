SEED=42
TRAIN_MODES="--first-mode 0 --last-mode 7"
# python run_experiments.py --method_type Baseline  $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type Finetune  $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type CompoNet  $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type PackNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type ProgNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type TvNet     $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type FuseNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type MaskNet  $TRAIN_MODES --seed $SEED --tag Baseline
python run_experiments.py --method_type CbpNet  $TRAIN_MODES --seed $SEED --tag Baseline

SEED=42
TRAIN_MODES="--first-mode 0 --last-mode 7"
export CUDA_VISIBLE_DEVICES=0
# python run_experiments.py --method_type Baseline  $TRAIN_MODES --seed $SEED --tag release
# python run_experiments.py --method_type Finetune  $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type ProgNet   $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type PackNet   $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type MaskNet  $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type CReLUs  $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type CompoNet  $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type CbpNet  $TRAIN_MODES --seed $SEED  --tag release
# python run_experiments.py --method_type CKA-RL   $TRAIN_MODES --seed $SEED  --tag release
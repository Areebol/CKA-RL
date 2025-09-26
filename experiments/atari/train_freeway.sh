SEED=42
TRAIN_MODES="--first-mode 0 --last-mode 7"
export CUDA_VISIBLE_DEVICES=0
# python run_experiments.py --method_type Finetune  $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type CompoNet  $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type PackNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type ProgNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type TvNet     $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type FuseNet   $TRAIN_MODES --seed $SEED --debug True
# python run_experiments.py --method_type MaskNet  $TRAIN_MODES --seed $SEED --tag Baseline
# python run_experiments.py --method_type CbpNet  $TRAIN_MODES --seed $SEED --tag Baseline
# python run_experiments.py --method_type Rewire  $TRAIN_MODES --seed $SEED --tag Baseline
# python run_experiments.py --method_type Rewire  $TRAIN_MODES --seed $SEED --debug
# python run_experiments.py --method_type CReLUs  $TRAIN_MODES --seed $SEED --tag Baseline
# python run_experiments.py --method_type Baseline  $TRAIN_MODES --seed $SEED --tag ExchangeTaskOrder_1
python run_experiments.py --method_type CompoNet  --first-mode 0 --last-mode 7 --seed 42 --tag ExchangeTaskOrder_3 --task_order 3
# python run_experiments.py --method_type FuseNet  --first-mode 0 --last-mode 7 --seed 42 --tag ExchangeTaskOrder_1 --fuse_actor --task_order 1


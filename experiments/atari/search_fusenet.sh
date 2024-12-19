SEED=42
TRAIN_MODES="--first-mode 2 --last-mode 7"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-3 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-3"
# python run_experiments.py --method_type FuseNet   --alpha_factor 1e-4 $TRAIN_MODES --seed $SEED --tag "AlphaFactor1e-4"
# TAT = theta + alpha tau 

# without global alpha
# fuse encoder and actor
# python run_experiments.py --method_type FuseNet \
#                           --alpha_learning_rate 2.5e-4 \
#                           --delta_theta_mode TAT \
#                           --alpha_factor 1e-3 \
#                           --fuse_encoder \
#                           --fuse_actor \
#                           --tag "FuseEncoder&Actor" \
#                           $TRAIN_MODES \
#                           --seed $SEED \

# only fuse encoder
# python run_experiments.py --method_type FuseNet \
#                           --alpha_learning_rate 2.5e-4 \
#                           --delta_theta_mode TAT \
#                           --alpha_factor 1e-3 \
#                           --fuse_encoder \
#                           --tag "FuseEncoderOnly" \
#                           $TRAIN_MODES \
#                           --seed $SEED \

# only fuse actor
python run_experiments.py --method_type FuseNet \
                          --alpha_learning_rate 2.5e-3 \
                          --delta_theta_mode TAT \
                          --fuse_actor \
                          --tag "FuseActorOnlyRandn2.5e-3" \
                          $TRAIN_MODES \
                          --seed $SEED \

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7
python run_experiments.py --algorithm componet --start-mode 0 --first-mode 0 --last-mode 0
python run_experiments.py --algorithm finetune --start-mode 0 --first-mode 0 --last-mode 0
python run_experiments.py --algorithm from-scratch --start-mode 0 --first-mode 0 --last-mode 0
python run_experiments.py --algorithm prog-net --start-mode 0 --first-mode 0 --last-mode 0
python run_experiments.py --algorithm packnet --start-mode 0 --first-mode 0 --last-mode 0

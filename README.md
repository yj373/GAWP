# Geometry-aware Weight Perturbation for Adversarial Training
This is the official implementation of the paper "Geometry-aware Weight Perturbation for Adversarial Training"

## Set up
The code is based on Pytorch. Before using the code, construct a conda environmet with
```
conda env create -f environment.yaml
```
Then, activate the environment, using
```
conda activate GAWP
```

## Example
For AT-GAWP, an example command is provided
```
python main_PGDAT.py --wp --gawp --ga-lambda -1.0 --wp-gamma 0.0125 --wp-warmup 5 --wp-threshold -1 --wp-K2 10 --seed 0 --model PreActResNet18 --data cifar10 --weight-decay 5e-4 --batch-size 128 --data-dir /data --epochs 200 --lr-schedule piecewise --lr-init 0.1 --lr-gamma 0.1 --ckpt-iters 50 --norm 'l_inf' --epsilon 8 --eval-epsilon 8 --pgd-alpha 2 --eval-pgd-alpha 2 --attack-iters 10 --eval-attack-iters 10 --save-dir results/WP/c10/GAWP_10
```
For AT-TRADES, an example command is provided
```
python main_TRADES.py --wp --gawp --ga-lambda -1.0 --wp-gamma 0.005 --wp-warmup 0 --wp-threshold -1 --wp-K2 10 --seed 0 --model PreActResNet18 --data cifar10 --weight-decay 5e-4 --batch-size 128 --data-dir /data --epochs 200 --lr-schedule piecewise --lr-init 0.1 --lr-gamma 0.1 --ckpt-iters 50 --norm 'l_2' --epsilon 128 --eval-epsilon 128 --pgd-alpha 15 --eval-pgd-alpha 15 --attack-iters 10 --eval-attack-iters 10 --save-dir results/WP/c10/TRADES_GAWP_L2_1
```
For AT-MART, an example command is provided
```
python main_MART.py --wp --gawp --ga-lambda -1.0 --wp-gamma 0.01 --wp-warmup 5 --wp-threshold -1 --wp-K2 10 --seed 0 --model PreActResNet18 --data cifar10 --weight-decay 5e-4 --batch-size 128 --data-dir /data --epochs 200 --lr-schedule piecewise --lr-init 0.1 --lr-gamma 0.1 --ckpt-iters 50 --norm 'l_2' --epsilon 128 --eval-epsilon 128 --pgd-alpha 15 --eval-pgd-alpha 15 --attack-iters 10 --eval-attack-iters 10 --save-dir results/WP/c10/MART_GAWP_L2
```

## Reference Code

- RWP: https://github.com/ChaojianYu/Robust-Weight-Perturbation
- MLCAT: https://github.com/ChaojianYu/Understanding-Robust-Overfitting
- TRADES: https://github.com/yaodongyu/TRADES/
- MART: https://github.com/YisenWang/MART
- AutoAttack: https://github.com/fra31/auto-attack



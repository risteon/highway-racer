# RACER with Highway Environment

<video src="results/best_policy_highway_eval_checkpoint_190000_episode_4.mp4" width="640" controls></video>

This repository integrates RACER (Risk-sensitive Actor Critic with Epistemic Robustness) with a continuous-action highway-env to train a distributional SAC.
The original RACER implementation from Kyle Stachowicz can be found at https://github.com/kylestach/epistemic-rl-release.


## Run training script

Setup a virtual env (tested with python 3.11):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training:
```bash
python -m racer.scripts.train_highway
```

You should see that the averaged speed reaches ~35 m/s at step 30k.

## Evaluation

Comparing standard SAC with the distributional SAC implementation from RACER shows better performance and training convergence.
![Quantitative results](results/quantitative_evaluation.png)

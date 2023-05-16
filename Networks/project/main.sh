#!/bin/bash
set -x
command="python main.py --activation_type hard --learning_rule 1 --n_epochs 20 --weight_init herman --bias_init nesbit --symmetric"
eval "$command"
command="python main.py --activation_type lif --learning_rule 0 --n_epochs 20 --weight_init herman --bias_init nesbit"
eval "$command"
command="python main.py --activation_type loihi --learning_rule 0 --n_epochs 20 --weight_init bengio --bias_init nesbit"
eval "$command"
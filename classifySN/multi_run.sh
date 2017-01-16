#!/bin/bash

python classify_model.py --filename '50-50-32-4' --quick_log_file 'log_50_50_32_4.csv' \
                        --n_hidden 200 --n_epochs 200 --batch_size 32 \
                        --n_layers 6 --b_per_epoch 200 \
                        --log_directory 'tmp/test/50_50_32_4'

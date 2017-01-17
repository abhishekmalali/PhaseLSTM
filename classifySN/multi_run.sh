#!/bin/bash

python classify_model.py --filename '100-50-32-5' --quick_log_file 'log_100_50_32_5.csv' \
                        --n_hidden 100 --n_epochs 50 --batch_size 32 \
                        --n_layers 5 --b_per_epoch 200 \
                        --log_directory 'tmp/test/100_50_32_5'

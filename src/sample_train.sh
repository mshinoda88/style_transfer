#!/bin/bash

#style=la_muse
#style=mirror
#style=yawara
style=lassen
#epochs=450
epochs=5
python train.py -s $style -o $style -e $epochs \
  >> sample_train.log 2<&1



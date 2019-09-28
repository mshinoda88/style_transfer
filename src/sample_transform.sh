#!/bin/bash

style=la_muse
#style=lassen
#style=mirror
#style=udnie
#style=yawara
input_image=../images/content/bean01.jpg
#input_image=../images/content/101.jpg
rate=0.1
#rate=0.05
python transform.py -i ${input_image} -s $style -b ${rate} -o bean -d ../images/output \
    >> sample_transform.log 2>&1


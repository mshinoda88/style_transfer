#!/bin/bash

function download_gd() {
  fileid=$1
  filename=$2
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${fileid}" -o ${filename}
}

fileid="16lKw4Z80NF0kf5PVzq3zXEM-FBrQ3wi1"
filename="wave_crop_weights.h5"
download_gd $fileid $filename

fileid="1ve3-e3B0MhyZMzX3Gvw2o8YhjBdqWS7m"
filename="udnie_weights.h5"
download_gd $fileid $filename

fileid="1_WuBGnjAX3f1AMTM3yy-_DzqvDus1iDs"
filename="mirror_weights.h5"
download_gd $fileid $filename

fileid="1LE-hnScPJmAXBUNik6OosFO4l43tnHIx"
filename="la_muse_weights.h5"
download_gd $fileid $filename

fileid="1f6xXfpBZ77qzD-qunfLVv0GeygQTIa1B"
filename="des_glaneuses_weights.h5"
download_gd $fileid $filename


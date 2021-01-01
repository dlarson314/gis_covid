#!/bin/bash

set -x

ffmpeg -framerate 10 -i frame%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output2048.mp4


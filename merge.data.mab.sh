#!/bin/bash - 
#======================================================
#
#          FILE: merge.data.mab.sh
# 
USAGE="./merge.data.mab.sh"
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: --- unknown
#         NOTES: ---
#        AUTHOR: |CHAO.TANG| , |chao.tang.1@gmail.com|
#  ORGANIZATION: 
#       CREATED: 07/11/17 09:42
#      REVISION: 1.0
#=====================================================
set -o nounset           # Treat unset variables as an error
. ~/Shell/functions.sh   # ctang's functions


OUTPUT=sarah_2.vld.sta.mab.csv


# input files:

file1=sarah_2.vld.series.8305.MAB.csv 
# by saraH_2.vld.sta.series.shareXY.py

file2=GEBA.station.gt.1mon.SA

#=================================================== 
echo "sta_ID mab" > $file1.T

space.sh $file1 > $file1.space
transpose.sh $file1.space >> $file1.T

merge.sh $file1.T $file2 > $OUTPUT.temp

space.sh $OUTPUT.temp > $OUTPUT.temp2

awk '{print $3","$2","$3","$4","$5","$6","$7","$3","$4","$5","$6","$7","$8","$9}' $OUTPUT.temp2 > $OUTPUT

rm $file1.* $OUTPUT.temp*

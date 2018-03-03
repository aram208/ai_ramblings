#! /bin/bash

# create the descriptor for the negatives (aka background) images
$ find ./negative_images -iname "*.jpg" > negatives.txt
#find ./originals/cropped_resized -iname "*.jpg" > positives.txt

$ opencv_createsamples -img ./originals/cropped_resized/1_cropped_v.jpg -bg negatives.txt -info info1/info1.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802
$ opencv_createsamples -img ./originals/cropped_resized/2_cropped_v.jpg -bg negatives.txt -info info2/info2.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802
$ opencv_createsamples -img ./originals/cropped_resized/3_cropped_v.jpg -bg negatives.txt -info info3/info3.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802
$ opencv_createsamples -img ./originals/cropped_resized/4_cropped_v.jpg -bg negatives.txt -info info4/info4.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802

# copy all from info1, info2 etc into info
$ cp -a info1/. info/

# merge info1, info2 etc into merged.txt, then remove them and rename merged to info.lst
$ cat info/info*.lst > info/merged.txt
$ rm -f info/*.lst
$ mv info/merged.txt info/info.lst

# then run this to generate the positive vector (I have about 5892 items in the info folder)
$ opencv_createsamples -info info/info.lst -num 5800 -w 20 -h 20 -vec positives.vec

# and finally the training (dont forget to have the 'data' folder) I have 1802 negatives
# to avoid the dreaded "no more positives" error make sure the following:
#-numPose is a samples count that is used to train each stage. Some already used
#samples can be filtered by each previous stage (ie recognized as background),
#but no more than (1 – minHitRate) * numPose on each stage.
#So vec-file has to contain >= (numPose + (numStages-1) * (1 – minHitRate) * numPose) + S,
#where S is a count of samples from vec-file that can be recognized as background right away.
#I hope it can help you to create vec-file of correct size and chose right numPos value.
# In plain english, use a number that is 80% or 90% of total positives
# !!! carefull with the last three parameters. It may prolong the training indefinitely
$ opencv_traincascade -data data -vec positives.vec -bg negatives.txt -numPos 5000 -numNeg 1800 -numStages 10 -w 20 -h 20 -minHitRate 0.999 -maxFalseAlarmRate 0.1 -maxWeakCount 1000

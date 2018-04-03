#! /bin/bash

# create the descriptor for the negatives (aka background) images
$ find ./negative_images -iname "*.jpg" > negatives.txt
#find ./originals/cropped_resized -iname "*.jpg" > positives.txt

$ opencv_createsamples -img ./originals/cropped_resized/1_cropped_v.jpg -bg negatives.txt -info info1/info1.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802

$ opencv_createsamples -img ./originals/cropped_resized/2_cropped_v.jpg -bg negatives.txt -info info2/info2.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802

$ opencv_createsamples -img ./originals/cropped_resized/3_cropped_v.jpg -bg negatives.txt -info info3/info3.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1802



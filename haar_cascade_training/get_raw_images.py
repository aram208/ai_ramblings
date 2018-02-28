# -*- coding: utf-8 -*-

import urllib.request
import cv2
import os

#def store_raw_images():
neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09403734'   
neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
pic_num = 677

if not os.path.exists('negative_images'):
    os.makedirs('negative_images')
    
for i in neg_image_urls.split('\n'):
    try:
        print(i)
        urllib.request.urlretrieve(i, "negative_images/"+str(pic_num)+".jpg")
        img = cv2.imread("negative_images/"+str(pic_num)+".jpg",cv2.IMREAD_GRAYSCALE)
        # should be larger than samples / pos pic (so we can place our image on it)
        resized_image = cv2.resize(img, (100, 100))
        cv2.imwrite("negative_images/"+str(pic_num)+".jpg",resized_image)
        pic_num += 1
        
    except Exception as e:
        print(str(e))
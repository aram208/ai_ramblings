# -*- coding: utf-8 -*-

import urllib.request
import cv2
import os

# some imaage_net ids
# n00523513 - athletics
# n09403734 - mountains
# n03521431 - gates and hinges
# n12992868 - fungus
# n07713895 - cabbage
# n04255899 - buildings
# n00017222 - plants
# n09287968 - geoformations
# n02898711 - bridges and spans
neg_images_link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02898711"
net_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
pic_num = 4603

if not os.path.exists('negative_images_color'):
    os.makedirs('negative_images_color')
    
for i in net_image_urls.split("\n"):
    try:
        print(i)
        urllib.request.urlretrieve(i, "negative_images_color/" + str(pic_num) + ".jpg")
        # img = cv2.imread("negative_images/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
        # make sure the size is larger than the samples that we will put inside
        img = cv2.imread("negative_images_color/" + str(pic_num) + ".jpg")
        resized_image = cv2.resize(img, (100, 100))
        cv2.imwrite("negative_images_color/" + str(pic_num) + ".jpg", resized_image)
        pic_num += 1
        
    except Exception as e:
        print(str(e))

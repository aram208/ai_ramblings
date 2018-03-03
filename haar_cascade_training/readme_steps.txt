source activate virtual_platform
cd Projects/AI/shared/ai_ramblings/haar_cascade_training/

# run the get raw images to download the negatives

# remove the blank/error pages

# if necessary, generate the descriptor for negative_images
find ./neg -iname "*.jpg" > negatives.txt

# generate the samples using ./create_positives.sh

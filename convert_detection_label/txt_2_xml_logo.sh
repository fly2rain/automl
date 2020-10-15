# 1. txt 2 xml label converting
python3 txt_2_xml.py \
        -format "yolo" \
        -src  /media/fyzhu/data2T_1/backup_dataset/logo_detection/backup_data_Mar19_2020/20191128_manual_collected_backup \
        -dst ~/Documents/20191128_manual_collected_backup \
        -class_list  Logo.names

# 2. xml & image 2 tfrecord
data_path="/media/fyzhu/data2T_1/backup_dataset_public/voc2012"
export PYTHONPATH=".:$PYTHONPATH"
python3  create_pascal_tfrecord.py  \
    --data_dir=${data_path}/VOCdevkit --year=VOC2012 \
    --output_path=${data_path}/tfrecord/pascal
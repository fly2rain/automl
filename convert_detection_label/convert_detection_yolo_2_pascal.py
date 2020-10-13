import os
import xml.etree.ElementTree as ET
import random
import PIL.Image
from .utils_fyzhu import OsProcess, GetFileLists

input_dataset_path = 'ItemRecognition'
input_datasets = os.listdir(input_dataset_path)
input_datasets = [f'{input_dataset_path}/{d}/images' for d in input_datasets if d.startswith('76051_')]
big_folders = input_datasets

output_dataset = 'shelf_detection_VOC_shrink_size'
VALID_SPLIT = 1.0
SHRINK_SIZE = 1024

print(big_folders)


def add_object(tree, name, bbox, last_obj):
    # Add an object to current xml tree

    obj = ET.Element('object')
    obj.text = '\n\t\t'
    if not last_obj:
        obj.tail = '\n\t'
    else:
        obj.tail = '\n'

    name_ele = ET.SubElement(obj, 'name')
    name_ele.text = name
    name_ele.tail = '\n\t\t'

    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Left'
    pose.tail = '\n\t\t'

    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = str(0)
    truncated.tail = '\n\t\t'

    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = str(0)
    difficult.tail = '\n\t\t'

    bndbox = ET.SubElement(obj, 'bndbox')
    bndbox.text = '\n\t\t\t'
    bndbox.tail = '\n\t'

    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(bbox[0])
    xmin.tail = '\n\t\t\t'

    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(bbox[1])
    ymin.tail = '\n\t\t\t'

    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(bbox[2])
    xmax.tail = '\n\t\t\t'

    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(bbox[3])
    ymax.tail = '\n\t\t'

    tree.append(obj)


def convert(size, box):
    # convert yolo box to voc box
    # size = imagesize(width,height), box = yolo_box(x,y,w,h)

    # box = float(box)
    box = [float(b) for b in box]
    voc_box = [0, 0, 0, 0]
    bboxW = box[2] * size[0]
    bboxH = box[3] * size[1]
    centerX = box[0] * size[0]
    centerY = box[1] * size[1]

    voc_box[0] = centerX - (bboxW / 2)
    voc_box[1] = centerY - (bboxH / 2)
    voc_box[2] = centerX + (bboxW / 2)
    voc_box[3] = centerY + (bboxH / 2)

    voc_box[0] = max(0, voc_box[0])
    voc_box[0] = min(size[0] - 2, voc_box[0])
    voc_box[1] = max(0, voc_box[1])
    voc_box[1] = min(size[1] - 2, voc_box[1])

    voc_box[2] = min(size[0] - 1, voc_box[2])
    voc_box[2] = max(0, voc_box[2])
    voc_box[3] = min(size[1] - 1, voc_box[3])
    voc_box[3] = max(0, voc_box[3])

    # round to int
    voc_box = [int(round(vb)) for vb in voc_box]
    # voc_box = map(int,map(round,voc_box))

    return voc_box  # (xmin,ymin,xmax,ymax)


def convert_annotation(IM_HEIGHT, IM_WIDTH, txt_path, xml_output_path):
    # image_dim = cv2.imread(image_id).shape
    image_dim = list([IM_WIDTH, IM_HEIGHT])
    # print(txt_path)
    in_file_label = open(txt_path)
    labels = in_file_label.readlines()
    # print(labels)

    # Read example xml and modify it
    example = open('example.xml')
    tree = ET.parse(example)
    root = tree.getroot()
    file_name = root.find('filename')
    file_name.text = xml_output_path.split('/')[-1].replace('.xml', '.jpg')
    segmented = root.find('segmented')
    segmented.tail = '\n\t'
    size = root.find('size')
    image_width = size.find('width')
    image_width.text = str(image_dim[0])
    image_height = size.find('height')
    image_height.text = str(image_dim[1])

    label_idx = 0
    last = False
    # Add objects in txt to xml
    for label in labels:
        label = label.split()
        # name = str(label[0])
        name = str(1)
        bbox = label[1:]

        if label_idx == (len(labels) - 1):
            last = True
        bbox_yolo = convert(image_dim, bbox)
        add_object(root, name, bbox_yolo, last)
        label_idx += 1

    tree.write(xml_output_path)


if not os.path.exists(output_dataset):
    os.mkdir(output_dataset)
if not os.path.exists(output_dataset + '/' + 'VOCdevkit'):
    os.mkdir(output_dataset + '/' + 'VOCdevkit')
if not os.path.exists(output_dataset + '/' + 'VOCdevkit' + '/' + 'VOC2012'):
    os.mkdir(output_dataset + '/' + 'VOCdevkit' + '/' + 'VOC2012')
if not os.path.exists(output_dataset + '/' + 'VOCdevkit' + '/' + 'VOC2012' + '/' + 'ImageSets'):
    os.mkdir(output_dataset + '/' + 'VOCdevkit' + '/' + 'VOC2012' + '/' + 'ImageSets')

output_image_folder = output_dataset + '/' + 'VOCdevkit/' + 'VOC2012/' + 'JPEGImages'
if not os.path.exists(output_image_folder):
    os.mkdir(output_image_folder)

output_annotation_folder = output_dataset + '/' + 'VOCdevkit/' + 'VOC2012/' + 'Annotations'
if not os.path.exists(output_annotation_folder):
    os.mkdir(output_annotation_folder)

output_imagesets_folder = output_dataset + '/' + 'VOCdevkit/' + 'VOC2012/' + 'ImageSets/' + 'Main/'
if not os.path.exists(output_imagesets_folder):
    os.mkdir(output_imagesets_folder)

i = 0
train_list = []
val_list = []
for folder in big_folders:
    print("Start Processing Folder: ", folder)
    image_paths_raw = os.listdir(folder)
    # image_paths = [folder+'/'+image_path for image_path in image_paths if image_path.endswith('.jpg')
    # and '.DS_Store' not in image_path and '._' not in image_path]
    image_paths = []
    for image_path in image_paths_raw:
        if '.DS_Store' in image_path and '._' in image_path:
            continue
        if image_path.endswith('.jpg') or image_path.endswith('.JPG') or image_path.endswith('.png'):
            image_paths.append(folder + '/' + image_path)
    # txt_path = folder

    for image_path in image_paths:
        image = PIL.Image.open(image_path)
        IM_HEIGHT = image.size[1]
        IM_WIDTH = image.size[0]
        # shrink image size if the larger side is greater than SHRINK_SIZE

        if max(IM_HEIGHT, IM_WIDTH) > SHRINK_SIZE:
            image_scale_y = (SHRINK_SIZE * 1.0) / (IM_HEIGHT * 1.0)
            image_scale_x = (SHRINK_SIZE * 1.0) / (IM_WIDTH * 1.0)
            image_scale = min(image_scale_x, image_scale_y)

            scaled_height = int(round(IM_HEIGHT * image_scale))
            scaled_width = int(round(IM_WIDTH * image_scale))

            image = image.resize((scaled_width, scaled_height), PIL.Image.BILINEAR)

            IM_HEIGHT = image.size[1]
            IM_WIDTH = image.size[0]

        txt_path = image_path.replace('/images/', '/labels/')
        txt_path = os.path.splitext(txt_path)[0] + '.txt'
        # txt_path = txt_path.replace('.jpg','.txt')
        # print('read')
        if not os.path.exists(txt_path):
            print(image_path, ' not labeled!!')
            continue
        if random.uniform(0, 1) < VALID_SPLIT:
            # training data
            image.save(os.path.join(output_image_folder, (str(i) + '.jpg')))

            # txt_path = image_path.replace('.jpg','.txt')
            xml_output_path = os.path.join(output_annotation_folder, (str(i) + '.xml'))
            xml_output_path = xml_output_path.replace('\\', '/')
            convert_annotation(IM_HEIGHT, IM_WIDTH, txt_path, xml_output_path)
            train_list.append(i)
        else:
            # valid_data
            image.save(os.path.join(output_image_folder, (str(i) + '.jpg')))

            # txt_path = image_path.replace('.jpg','.txt')
            xml_output_path = os.path.join(output_annotation_folder, (str(i) + '.xml'))
            xml_output_path = xml_output_path.replace('\\', '/')
            convert_annotation(IM_HEIGHT, IM_WIDTH, txt_path, xml_output_path)
            val_list.append(i)
        i += 1
        if i % 100 == 0:
            print(i)

with open(output_imagesets_folder + '/aeroplane_train.txt', 'w', newline='') as f1:
    for path in train_list:
        if os.path.exists(output_image_folder + '/' + str(path) + '.jpg'):
            f1.write(str(path) + os.linesep)

with open(output_imagesets_folder + '/aeroplane_val.txt', 'w', newline='') as f2:
    for path in val_list:
        if os.path.exists(output_image_folder + '/' + str(path) + '.jpg'):
            f2.write(str(path) + os.linesep)

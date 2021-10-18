import sys

def add_detected_annotations(id_annotations: list, annotations: list, annotation_id: int, img_id: int):
    # append info to detected `annotations`, add annotation and count `annotation_id`
    for id_annotation in id_annotations:
        # append to annotation info
        id_annotation['image_id'] = img_id
        id_annotation['id'] = annotation_id   
        # add annotation
        annotations.append(id_annotation)
        annotation_id += 1

def add_detected_categories(id_categories: list, categories: list):
    # append _new_ category to present annotation categories (each category is in categories just once)
    for id_category in id_categories:
        category = next((category for category in categories if category['id'] == id_category['id']), None)
        if category is None:
            categories.append(id_category)

def add_image_info(coco_annotation_data: dict, coco_annotation_data_fpath: str, images: list, img_id: int):
    # appends image with `img_id` if found in coco_annotation_data (each image is in images just once)
    image = next((image for image in coco_annotation_data['images'] if image['id'] == img_id), None)
    if image is None:
        print('Error: image with id {} not found in {}'.format(img_id, coco_annotation_data_fpath), file=sys.stderr)
        print('Abort detection.')
        sys.exit(1)

    images.append(image)
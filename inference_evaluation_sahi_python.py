#from detectron2.config import get_cfg
import os
import torch
import numpy as np
import cv2
import json

from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, COCOEvaluator
from detectron2.modeling import build_model

from detectron2.checkpoint import DetectionCheckpointer

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

#import DatasetCatalog, MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog

#import logger
from detectron2.utils.logger import setup_logger
setup_logger()

from utils.structure_utils import map1_cr, map2_cr, structure1_cr, structure2_cr, single_syn_cr

#Import custom dataset mapper for augmentation
from data_augmentation.augmentation_datasetmapper import AugmentDatasetMapper
from data_augmentation.augmentations import create_augmentations
from data_augmentation.augmentations_utils import create_bbox_hflip_indices, create_bbox_vflip_indices

#import visualizer
from detectron2.utils.visualizer import Visualizer

import sahi
from sahi import AutoDetectionModel
from sahi.predict import predict, get_prediction, get_sliced_prediction


def main():

    """
    num_classes = 515
    
    # Create map1
    map1_cr(ontology_jsonld_path='structures/PIDOntology.jsonld')

    # Create map2
    map2_cr()

    # Create structure2
    structure2_cr(ontology_owl_path='structures/PIDOntology.owl',
                    num_classes=num_classes+1)

    # Create structure1
    structure1_cr()

    # Create single_syn
    single_syn_cr()

    # Register hierarchical roi head
    from hierarchical_roi_heads.hierarchical_roi_heads import register_roi_head
    register_roi_head()
    """
    

    #model_path = r"W:\staff-umbrella\piresearchStudentsShared\Winston_Oudshoorn\05_Data\Experiments\8. Exp_SWMI_baseline_2\results\output\model_final.pth"
    #config_path = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

    #train_dataset_img_path = r"W:\staff-umbrella\piresearchStudentsShared\Winston_Oudshoorn\05_Data\Data\SWMI\images"
    #train_dataset_annot_path = r"W:\staff-umbrella\piresearchStudentsShared\Winston_Oudshoorn\05_Data\Data\SWMI\SWMI_baseline\train.json"

    # args
    config_path = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"

    # model
    model_path = r"C:\Users\wouds\Documents\GitHub\Flowsheet_Digitization\experiments\output\ASPEN_1\output\model_final.pth"

    # test dataset
    image_path = r"C:\Users\wouds\Documents\GitHub\Flowsheet_Digitization\data\ASPEN_1\images"
    json_path = r"C:\Users\wouds\Documents\GitHub\Flowsheet_Digitization\data\ASPEN_1\test.json"

    train_dataset_annot_path = json_path #r"C:\Users\wouds\Documents\GitHub\Flowsheet_Digitization\data\ASPEN_1\train.json"
    train_dataset_img_path = image_path #r"C:\Users\wouds\Documents\GitHub\Flowsheet_Digitization\data\ASPEN_1\images"



    
    try:
        register_coco_instances(name='train', metadata={}, json_file=train_dataset_annot_path, image_root=train_dataset_img_path)
    except:
        DatasetCatalog.remove('train')
        MetadataCatalog.remove('train')
        register_coco_instances(name='train', metadata={}, json_file=train_dataset_annot_path, image_root=train_dataset_img_path)
    dataset = DatasetCatalog.get('train')
    metadata = MetadataCatalog.get('train')
    category_names = metadata.thing_classes
    category_mapping = {
                            str(ind): category_name for ind, category_name in enumerate(category_names)
                        }
    

    # alternative way to get the categories from the json
    with open(train_dataset_annot_path) as f:
        train_dataset_annot = json.load(f)

    # get the categories from the json
    categories = train_dataset_annot['categories']

    # set up category mapping
    #category_mapping = {}
    #for category in categories:
    #    category_mapping[str(category['id'])] = category['name']


    result = predict(
        #detection_model=model,
        model_type='detectron2',
        model_path=model_path,
        model_config_path=config_path,
        model_confidence_threshold=0.2,
        model_device="cuda", # or 'cuda:0'
        source=image_path,
        model_category_mapping=category_mapping,
        no_standard_prediction = False,
        no_sliced_prediction = True,
        novisual = False,
        verbose = 1,
        return_dict = True,
        project="output/sahi",
        name="test",
        visual_bbox_thickness = 1,
        visual_text_size = 0.4,
        visual_text_thickness = 1,
        visual_export_format = "png",
        dataset_json_path = json_path,
    )
"""
    result2 = predict(
        #detection_model=model,
        model_type='detectron2',
        model_path=model_path,
        model_config_path=config_path,
        model_confidence_threshold=0.5,
        model_device="cuda", # or 'cuda:0'
        source=image_path,
        model_category_mapping=category_mapping,
        no_standard_prediction = False,
        no_sliced_prediction = False,
        novisual = False,
        verbose = 2,
        return_dict = True,
        slice_height = 512,
        slice_width = 512,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        project="output/sahi",
        name="result_slice_512_postprocess_False_0.5",
        visual_bbox_thickness = 1,
        visual_text_size = 0.4,
        visual_text_thickness = 1,
        visual_hide_labels = False,
        visual_hide_conf = False,
        visual_export_format = "png",
        dataset_json_path = json_path,
    )

    result3 = predict(
        #detection_model=model,
        model_type='detectron2',
        model_path=model_path,
        model_config_path=config_path,
        model_confidence_threshold=0.5,
        model_device="cuda", # or 'cuda:0'
        source=image_path,
        model_category_mapping=category_mapping,
        no_standard_prediction = False,
        no_sliced_prediction = False,
        novisual = False,
        verbose = 2,
        return_dict = True,
        slice_height = 1024,
        slice_width = 1024,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        project="output/sahi",
        name="result_slice_1024_postprocess_False_0.5",
        visual_bbox_thickness = 1,
        visual_text_size = 0.4,
        visual_text_thickness = 1,
        visual_hide_labels = False,
        visual_hide_conf = False,
        visual_export_format = "png",
        dataset_json_path = json_path,
    )


    result4 = predict(
        #detection_model=model,
        model_type='detectron2',
        model_path=model_path,
        model_config_path=config_path,
        model_confidence_threshold=0.5,
        model_device="cuda", # or 'cuda:0'
        source=image_path,
        model_category_mapping=category_mapping,
        no_standard_prediction = False,
        no_sliced_prediction = False,
        novisual = False,
        verbose = 2,
        return_dict = True,
        slice_height = 512,
        slice_width = 512,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=True,
        project="output/sahi",
        name="result_slice_512_postprocess_True_0.5",
        visual_bbox_thickness = 1,
        visual_text_size = 0.4,
        visual_text_thickness = 1,
        visual_hide_labels = False,
        visual_hide_conf = False,
        visual_export_format = "png",
        dataset_json_path = json_path,
    )

    result3 = predict(
        #detection_model=model,
        model_type='detectron2',
        model_path=model_path,
        model_config_path=config_path,
        model_confidence_threshold=0.5,
        model_device="cuda", # or 'cuda:0'
        source=image_path,
        model_category_mapping=category_mapping,
        no_standard_prediction = False,
        no_sliced_prediction = False,
        novisual = False,
        verbose = 2,
        return_dict = True,
        slice_height = 1024,
        slice_width = 1024,
        overlap_height_ratio = 0.1,
        overlap_width_ratio = 0.1,
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=True,
        project="output/sahi",
        name="result_slice_1024_postprocess_True_0.5",
        visual_bbox_thickness = 1,
        visual_text_size = 0.4,
        visual_text_thickness = 1,
        visual_hide_labels = False,
        visual_hide_conf = False,
        visual_export_format = "png",
        dataset_json_path = json_path,
    )
"""

"""
    # results.json has an issue with the category mapping, so we fix it here
# add 1 to each category id
with open(results_json_path) as f:
    results_json = json.load(f)

# results_json structure = [{image_id, category_id, bbox, score}, ...]
for i in range(len(results_json)):
    results_json[i]['category_id'] = int(results_json[i]['category_id']) + 1

f.close()

# save the fixed results_json to a different file
results_json_path_2 = os.path.join(".", "output/sahi", output_name, "result_fixed.json")
with open(results_json_path_2, 'w') as f:
    json.dump(results_json, f)

f.close()

output_directory = os.path.join(path, "eval")
evaluate(dataset_json_path=json_path, result_json_path=results_json_path_2, out_dir=output_directory, 
    type="bbox", classwise=True, max_detections=500, iou_thrs=None, areas=[1024, 9216, 10000000000], return_dict=True
    )
"""


if __name__ == '__main__':
    main()
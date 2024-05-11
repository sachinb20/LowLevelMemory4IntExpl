import copy
import json
import os
from pathlib import Path
import random
import pickle
import warnings
import cv2
import math
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import trange
import argparse

from conceptgraph.utils.ai2thor import (
    get_agent_pose_from_event, 
    get_camera_pose_from_event, 
    get_top_down_frame,
    compute_intrinsics, 
    compute_pose, 
    get_scene, 
    sample_pose_uniform, 
    sample_pose_random
)
from conceptgraph.ai2thor.rearrange import rearrange_objects

NOT_TO_REMOVE = [
    "Wall",
    "Floor",
    "Window",
    "Doorway",
    "Room",
]


def generate_obs_from_poses(
    controller,
    K,
    sampled_poses,
    save_root,
    depth_scale=1000.0,
):
    color_path_temp = save_root + "/color/{:06d}.png"
    depth_path_temp = save_root + "/depth/{:06d}.png"
    instance_path_temp = save_root + "/instance/{:06d}.png"
    pose_path_temp = save_root + "/pose/{:06d}.txt"

    intrinsics_path = save_root + "/intrinsics.txt"
    obj_meta_path = save_root + "/obj_meta.json"
    color_to_object_id_path = save_root + "/color_to_object_id.pkl"
    object_id_to_color_path = save_root + "/object_id_to_color.pkl"
    video_save_path = save_root + "/rgb_video.mp4"

    # Generate and save images
    frames = []
    for i in trange(len(sampled_poses)):
        pose = sampled_poses[i]
        # Teleport the agent to the position and rotation
        
        # limit the horizon to [-30, 60]
        horizon = pose["horizon"]
        horizon = max(min(horizon, 60-1e-6), -30+1e-6)
        
        event = controller.step(
            action="Teleport",
            position=pose["position"],
            rotation=pose["rotation"],
            horizon=horizon,
            standing=pose["standing"],
            forceAction=True,
        )

        if not event.metadata["lastActionSuccess"]:
            # raise Exception(event.metadata["errorMessage"])

            # Seems that the teleportation failures are based on position. 
            # Once it fails on a position, it will fail on all orientations.
            # Therefore, we can simply skip these failed trials. 
            print("Failed to teleport to the position.", pose["position"], pose["rotation"])
            continue

        color = np.asarray(event.frame).copy()
        depth = np.asarray(event.depth_frame).copy()
        instance = np.asarray(event.instance_segmentation_frame).copy()

        # Compute the agent and camera pose. They are different!
        agent_pose = get_agent_pose_from_event(event)
        camera_pose = get_camera_pose_from_event(event)

        color_path = color_path_temp.format(i)
        depth_path = depth_path_temp.format(i)
        instance_path = instance_path_temp.format(i)
        pose_path = pose_path_temp.format(i)

        os.makedirs(os.path.dirname(color_path), exist_ok=True)
        imageio.imwrite(color_path, color)
        
        if args.save_video:
            frames.append(color)

        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        # Cut off the depth at 15 meters 
        # some points are outside the house are handled later. 
        depth[depth > 15] = 0
        depth_png = np.round(depth * depth_scale).astype(np.uint16)
        imageio.imwrite(depth_path, depth_png)
        
        os.makedirs(os.path.dirname(instance_path), exist_ok=True)
        imageio.imwrite(instance_path, instance)

        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        np.savetxt(pose_path, camera_pose)

    np.savetxt(intrinsics_path, K)
    
    # Save the objects information
    # Since we do not change the state, we can simply use the last event.
    # but the `visible` property does vary across frames, and thus they are not indicative in testing. 
    obj_meta = controller.last_event.metadata["objects"]
    with open(obj_meta_path, "w") as f:
        json.dump(obj_meta, f)
    
    # Save the color from/to object id mapping - they are global and constant across all frames/events. 
    # They are  tupled-indexed dict, and thus cannot be saved as JSON files. 
    color_to_object_id = event.color_to_object_id
    with open(color_to_object_id_path, "wb") as f:
        pickle.dump(color_to_object_id, f)
        
    object_id_to_color = event.object_id_to_color
    with open(object_id_to_color_path, "wb") as f:
        pickle.dump(object_id_to_color, f)
        
    if args.save_video:
        imageio.mimsave(video_save_path, frames, fps=20)
        print("Saved video to", video_save_path)

def sample_pose_from_file(traj_file):
    # Load the trajectory file (json)
    with open(traj_file, "r") as f:
        traj = json.load(f)

    sampled_poses = []
    for log in traj["agent_logs"]:
        sampled_poses.append(
            {
                "position": log["position"],
                "rotation": log["rotation"],
                "horizon": log["cameraHorizon"],
                "standing": log["isStanding"],
            }
        )

    return sampled_poses

def is_removeable(obj, level: int):
    if level == 1: # all objects except those in NOT_TO_REMOVE
        return obj['objectType'] not in NOT_TO_REMOVE
    elif level == 2: # objects that are pickupable or moveable
        return obj['pickupable'] or obj['moveable']
    elif level == 3: # objects that are pickupable
        return obj['pickupable']

def randomize_scene(args, controller) -> list[str]|None:
    '''
    Since we want to keep track of which objects are removed, but it is not done in ai2thor
    So we will keep of a list of object ids that are kept in the scene. 
    if no object is removed from the scene, then return None.
    '''
    if args.randomize_lighting:
        controller.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False,
        )
        
    if args.randomize_material:
        controller.step(
            action="RandomizeMaterials",
            useTrainMaterials=None,
            useValMaterials=None,
            useTestMaterials=None,
            inRoomTypes=None
        )
        
    # Randomly remove objects
    obj_list = controller.last_event.metadata["objects"]
    removed_object_ids = []
    if args.randomize_remove_ratio > 0.0:
        print("Before randomization, there are {} objects in the scene".format(len(obj_list)))
        for obj in obj_list:
            if is_removeable(obj, args.randomize_remove_level) and \
                random.random() < args.randomize_remove_ratio:
                controller.step(
                    action="DisableObject",
                    objectId=obj['objectId'],
                )
                removed_object_ids.append(obj['objectId'])
        print("After randomization, there are {} objects in the scene".format(
            len(controller.last_event.metadata["objects"])
        ))
    
    # Randomly move objects
    starting_poses, target_poses = None, None
    if args.randomize_move_pickupable_ratio > 0.0 or args.randomize_move_moveable_ratio > 0.0:
        starting_poses, target_poses = rearrange_objects(
            controller = controller,
            pickupable_move_ratio = args.randomize_move_pickupable_ratio,
            moveable_move_ratio = args.randomize_move_moveable_ratio,
            reset = False,
        )

    randomization_log = {
        "removed_object_ids": removed_object_ids,
        "starting_poses": starting_poses,
        "target_poses": target_poses,
        "randomize_lighting": args.randomize_lighting,
        "randomize_material": args.randomize_material,
    }

    return randomization_log

def randomize_scene_from_log(controller, randomization_log):
    if randomization_log['randomize_lighting']:
        warnings.warn("randomize_lighting from log file is not implemented yet")
    if randomization_log['randomize_material']:
        warnings.warn("randomize_material from log file is not implemented yet")
        
    # Remove some objects
    removed_object_ids = randomization_log['removed_object_ids']
    if len(removed_object_ids) > 0:
        for obj_id in removed_object_ids:
            event = controller.step(
                action="DisableObject",
                objectId=obj_id,
            )
            if not event.metadata['lastActionSuccess']:
                warnings.warn("Failed to remove object {}".format(obj_id))
                print(event.metadata['errorMessage'])
                
    # Set object poses
    target_poses = randomization_log['target_poses']
    if target_poses is not None:
        event = controller.step(
            action="SetObjectPoses",
            objectPoses=target_poses
        )
        if not event.metadata['lastActionSuccess']:
            warnings.warn("Failed to set object poses")
            print(event.metadata['errorMessage'])

def load_or_randomize_scene(args, controller):
    randomization_file_path = args.save_root + "/randomization.json"
    
    if os.path.exists(randomization_file_path):
        with open(randomization_file_path, "r") as f:
            randomization_log = json.load(f)
        randomize_scene_from_log(controller, randomization_log)
        print("Loaded Randomization from {}".format(randomization_file_path))
    else:
        randomization_log = randomize_scene(args, controller)
        with open(randomization_file_path, "w") as f:
            json.dump(randomization_log, f)
        print("Created randomization and saved to {}".format(randomization_file_path))
            
    return randomization_log

import math
from typing import Dict, List
import json
import open_clip
import torch
import torchvision

from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption

try: 
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "Tag2Text")
EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
RAM_PATH = os.path.join(GSA_PATH, "recognize-anything")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(EFFICIENTSAM_PATH)
sys.path.append(RAM_PATH)
try:
    from ram.models.tag2text import tag2text
    from ram.models.ram import ram as rm
    from ram import inference
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")



def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes



def create_scene_graph(objects):
    scene_graph = {}
    scene_graph["Agent"]={  
            "position": {
      "x": -0.34021228551864624,
      "y": 0.9094499349594116,
      "z": 1.0938019752502441
    },
        "center": {
      "x": -0.34021228551864624,
      "y": 0.9094499349594116,
      "z": 1.0938019752502441
    },

            "BoundingBox": [
      [
        -2.0334811210632324,
        1.7936310768127441,
        -0.2726714611053467
      ],
      [
        -2.0334811210632324,
        1.7936310768127441,
        -1.2774642705917358
      ],
      [
        -2.0334811210632324,
        -0.008308768272399902,
        -0.2726714611053467
      ],
      [
        -2.0334811210632324,
        -0.008308768272399902,
        -1.2774642705917358
      ],
      [
        -2.7560410499572754,
        1.7936310768127441,
        -0.2726714611053467
      ],
      [
        -2.7560410499572754,
        1.7936310768127441,
        -1.2774642705917358
      ],
      [
        -2.7560410499572754,
        -0.008308768272399902,
        -0.2726714611053467
      ],
      [
        -2.7560410499572754,
        -0.008308768272399902,
        -1.2774642705917358
      ]
    ],
            "parentReceptacles": ["Floor|+00.00|+00.00|+00.00"],
            "ObjectState": None
            }
    

    OBJECT_LIST = []
    for obj in objects:
        obj_id = obj["objectId"]
        aabb = obj["objectOrientedBoundingBox"]["cornerPoints"] if obj["pickupable"] else obj["axisAlignedBoundingBox"]["cornerPoints"]

        if obj["openable"]:
            if obj["isOpen"]:
                object_state = "Open"
            else:
                object_state = "Closed"
        else:
            object_state = None
        
        scene_graph[obj_id] = {
            "position": obj["position"],
            "center": obj["axisAlignedBoundingBox"]["center"],
            "BoundingBox": aabb,
            "parentReceptacles": obj["parentReceptacles"],
            "ObjectState": object_state
        }
        OBJECT_LIST.append(obj["objectType"])

    file_path = 'object_list.txt'

    with open(file_path, 'w') as file:
        file.write(','.join(map(str, OBJECT_LIST)))

    return scene_graph

def update_scene_graph(scene_graph,action,obj_id,recept_id):

    if action == "Pickup":
        scene_graph[obj_id]['parentReceptacles'] = ["Agent"]

    elif action == "Putdown":
        scene_graph[obj_id]['parentReceptacles'] = [recept_id]

    elif action == "Open":
        scene_graph[obj_id]['ObjectState'] = "Open"

    elif action == "Close":
        scene_graph[obj_id]['ObjectState'] = "Close"

    elif action == "Navigate":
        scene_graph = scene_graph

    return scene_graph


def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    out = reachable_positions[0]
    min_distance = float('inf')
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out

def find_keys(input_key, data):
    input_key = input_key.lower()
    matching_keys = []
    for key in data.keys():
        if key.lower().startswith(input_key):
            matching_keys.append(key)
    return matching_keys

def get_angle_and_closest_position(controller, object_type, scene_graph):
    # Extracting object and agent positions
    # types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    # assert object_type in types_in_scene
    # # print(types_in_scene)
    # obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == object_type)

    keys = find_keys(object_type, scene_graph)
    object_id = keys[0]                       #Choose first key
    obj_position = calculate_object_center(scene_graph[object_id]['BoundingBox'])

    # Save the reachable positions of the scene to a file
    reachable_positions = controller.step(
        action="GetReachablePositions", raise_for_failure=True
    ).metadata["actionReturn"]

    
    closest = closest_position(obj_position, reachable_positions)

    target_obj = controller.last_event.metadata["objects"][0]
    obj_x = target_obj["position"]["x"]
    obj_z = target_obj["position"]["z"]

    agent_position = controller.last_event.metadata["agent"]["position"]
    agent_x = agent_position["x"]
    agent_z = agent_position["z"]

    delta_x = obj_x - agent_x
    delta_z = obj_z - agent_z
    angle_rad = math.atan2(delta_z, delta_x)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, closest, object_id



def euclidean_distance(pos1, pos2):
    # print(pos1)
    # print(pos2)
    return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2 + (pos1['z'] - pos2['z'])**2)

def calculate_object_center(bounding_box):
    x_coords = [point[0] for point in bounding_box]
    y_coords = [point[1] for point in bounding_box]
    z_coords = [point[2] for point in bounding_box]
    center = {
        'x': sum(x_coords) / len(bounding_box),
        'y': sum(y_coords) / len(bounding_box),
        'z': sum(z_coords) / len(bounding_box)
    }
    return center

def find_closest_items(agent_position, scene_graph, num_items=5):
    distances = {}
    for obj_id, obj_data in scene_graph.items():
        # obj_position = obj_data['position']
        obj_aabb = obj_data['BoundingBox']
        obj_center = calculate_object_center(obj_aabb)
        # Adjusting object position relative to the agent
        obj_position_global = {
            'x': obj_center['x'] ,
            'y': obj_center['y'] ,
            'z': obj_center['z'] 
        }
        # Calculate distance from agent to object
        distance = euclidean_distance(agent_position, obj_position_global)
        distances[obj_id] = distance
    # Sort distances and return the closest items
    closest_items = sorted(distances.items(), key=lambda x: x[1])[:num_items]
    return closest_items


def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def get_sam_segmentation_from_point_and_box(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray,input_point: np.ndarray,input_label: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box,
                multimask_output=True,
            )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
    if variant == "sam":
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model
        MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError
    

def get_mask(bgr_frame):

    image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    sam_variant = "sam"
    sam_predictor = get_sam_predictor(sam_variant, args.device)
    mask = get_sam_segmentation_from_xyxy(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=np.array([[260,280,380,400]])
        )
    
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Create a white background
    # white_bg = np.ones_like(bgr_frame) * 255

    # # Draw contours on the white background
    # cv2.drawContours(white_bg, contours, -1, (0, 0, 0), thickness=1)

    # # Save the image
    # cv2.imwrite("masked.jpg", white_bg)

    image_np = np.array(image_rgb)
    print(mask.shape)
    print(np.shape(image_np))
    mask = mask[0]
    print(mask.shape)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)

    black_image.save("saved_image.png")

    return None    

def get_mask_with_pointprompt(bgr_frame):

    image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    sam_variant = "sam"
    sam_predictor = get_sam_predictor(sam_variant, args.device)
    mask = get_sam_segmentation_from_point_and_box(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=np.array([[230,240,410,480]]),
                input_point = np.array([[320, 340]]),
                input_label = np.array([1])
        )
    
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Create a white background
    # white_bg = np.ones_like(bgr_frame) * 255

    # # Draw contours on the white background
    # cv2.drawContours(white_bg, contours, -1, (0, 0, 0), thickness=1)

    # # Save the image
    # cv2.imwrite("masked.jpg", white_bg)

    image_np = np.array(image_rgb)
    print(mask.shape)
    print(np.shape(image_np))
    mask = mask[0]
    print(mask.shape)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    # black_image = Image.fromarray(black_image)

    # black_image.save("pepper_shaker_masked.png")
    cv2.imwrite("pepper_shaker_masked.png",black_image)
    return black_image  

def tagging_module(bgr_frame):

    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)

    specified_tags='None'
    # load model
    tagging_model = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                            image_size=384,
                                            vit='swin_b',
                                            delete_tag_index=delete_tag_index)
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tagging_model.threshold = 0.64 
    tagging_model = tagging_model.eval().to(args.device)
    tagging_transform = TS.Compose([
                            TS.Resize((384, 384)),
                            TS.ToTensor(), 
                            TS.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                                     ])
    
    image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Convert to RGB color space
    image_pil = Image.fromarray(image_rgb)
    raw_image = image_pil.resize((384, 384))
    raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)

    res = inference.inference_tag2text(raw_image , tagging_model, specified_tags)
    caption=res[2]
    
    text_prompt=res[0].replace(' |', ',')

    return caption, text_prompt

def save_frame(controller,state):

    bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(state+'.jpg', bgr_frame)

    return bgr_frame

def shift_indices(arr):
    continuous_parts = []
    current_part = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_part.append(arr[i])
        else:
            continuous_parts.append(current_part)
            current_part = [arr[i]]

    print(current_part)
    print(continuous_parts)
    if continuous_parts == []:
      return current_part
    else:
      return np.concatenate((current_part, continuous_parts[0]))
    
def visible_state(controller,target_receptacle):
    visibility_states = []

    for angle in range(12):
        last_rot = controller.last_event.metadata["agent"]["rotation"]["y"]
        controller.step(
            action="RotateLeft",
            degrees=30
        )
        #In case agent is stuck while rotating
        if last_rot == controller.last_event.metadata["agent"]["rotation"]["y"]:
            print("mera yasu yasu")
            rewind_angle = 1*30
            return rewind_angle
        

        types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
        assert target_receptacle in types_in_scene
        # print(types_in_scene)
        obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_receptacle)
        print(obj['visible'])
        visibility_states.append(obj['visible'])

        save_frame(controller,target_receptacle+'/'+str(angle+30))
 
    print(visibility_states)

    return visibility_states

def perturb(controller):
    # Define a list of actions
    actions = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
    
    # Choose a random action
    action = random.choice(actions)
    
    # Execute the chosen action
    controller.step(action)
    return None

def rotate_angle(controller,target_receptacle):


    # # Find the indices of all True values in the cyclical array
    # true_indices = [i for i, val in enumerate(visibility_states) if val]

    # # Calculate the total number of True values
    # num_true = len(true_indices)

    # # Calculate the length of the array
    # array_length = len(visibility_states)

    # # Initialize the shifted array
    # shifted_visibility_states = [False] * array_length

    # # Shift the segments if necessary
    # if num_true > 0:
    #     first_true_index = true_indices[0]
    #     shift_amount = array_length - first_true_index
    #     for i, val in enumerate(visibility_states):
    #         if val:
    #             shifted_visibility_states[(i + shift_amount) % array_length] = True

    visibility_states = visible_state(controller,target_receptacle)

    #Not well written but returns 30 degree rewind incase of hitting obs during rotation
    if type(visibility_states)  == int:
        return visibility_states
    
    #check whether not visible then pertube the agent
    while all(not elem for elem in visibility_states):
        perturb(controller)
        visibility_states = visible_state(controller,target_receptacle)


    true_indices = [i for i, val in enumerate(visibility_states) if val]

    # Find the indices of all True values in the shifted array
    shifted_true_indices = shift_indices(true_indices)
    midpoint_index = (len(shifted_true_indices) - 1) // 2

    # Get the index of the middle True value
    middle_index = shifted_true_indices[midpoint_index]
    print(middle_index)
    # Calculate the angle needed to rewind the rotation to that position
    rewind_angle = (11-middle_index) * 30

    return rewind_angle

def save_images_diff_views(event, folder="thirdpartycamera_images", prefix="snapshot"):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for i, rgb_frame in enumerate(event.third_party_camera_frames):
        rgb_filename = os.path.join(folder, f"{prefix}_{i}_rgb.png")
        cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

        depth_frame = event.third_party_depth_frames[i]
        depth_filename = os.path.join(folder, f"{prefix}_{i}_depth.png")
        cv2.imwrite(depth_filename, depth_frame)

def update_position_rotation(position, rotation, x=0, y=0, z=0, theta_x=0, theta_y=0, theta_z=0):
    # Create new dictionaries to hold updated values
    new_position = position.copy()
    new_rotation = rotation.copy()
    
    # Update position
    new_position['x'] += x
    new_position['y'] += y
    new_position['z'] += z
    
    # Update rotation
    new_rotation['x'] += theta_x
    new_rotation['y'] += theta_y
    new_rotation['z'] += theta_z
    
    return new_position, new_rotation

def normalize_rgb(rgb_array):
    # Remove the alpha channel and normalize the values
    rgb_array = rgb_array[:, :, :3] / 255.0
    return rgb_array


def save_frames_as_dict(event, camera_intrinsic, cam2base, prefix):
    observations = {}

    # Extract RGB frames and depth frames
    rgb_frames = event.third_party_camera_frames
    depth_frames = event.third_party_depth_frames
    
    # Assuming points, colors, depths, mask, cam2base are available for each frame
    for i in range(len(rgb_frames)):
        depth_shape = depth_frames[i].shape
        mask = np.full(depth_shape, True)  # Create a mask array with True values
        
        frame_dict = {
            "position": None,
            "rgb": normalize_rgb(rgb_frames[i]),
            "depths": depth_frames[i],
            "mask": mask,
            "c2w": cam2base[i],
            "intrinsic": camera_intrinsic,
            "dist_coef": np.zeros(5),  # Array of size 5 filled with zeros
        }
        observations[f"wrist_{i}"] = frame_dict
    
    # Save observations as pickle file
    filename = f"observations_{prefix}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(observations, file)
    print(observations)
    return filename

def construct_transformation_matrix(pos_dict, rot_dict):
    # Extract translation components
    x, y, z = pos_dict['x'], pos_dict['y'], pos_dict['z']

    # Extract rotation angles and convert to radians
    theta_x = np.radians(rot_dict['x'])
    theta_y = np.radians(rot_dict['y'])
    theta_z = np.radians(rot_dict['z'])

    # Translation matrix
    translation_matrix = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Rotation matrices along x, y, and z axes
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])

    rotation_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    rotation_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine translation and rotation matrices
    transformation_matrix = np.matmul(np.matmul(np.matmul(translation_matrix, rotation_x), rotation_y), rotation_z)

    return transformation_matrix


def dict_difference(point1, point2):
    difference = {}
    for key in point1.keys():
        difference[key] = point1[key] - point2.get(key, 0)
    return difference

def transformation_matrix(pos_dict, rot_dict):
    x, y, z = pos_dict['x'], pos_dict['y'], pos_dict['z']

    # Extract rotation angles and convert to radians
    theta_x = np.radians(rot_dict['x'])
    theta_y = np.radians(rot_dict['y'])
    theta_z = np.radians(rot_dict['z'])

    # Rotation matrices
    Rx = [[1, 0, 0, 0],
          [0, np.cos(theta_x), -np.sin(theta_x), 0],
          [0, np.sin(theta_x), np.cos(theta_x), 0],
          [0, 0, 0, 1]]

    Ry = [[np.cos(theta_y), 0, np.sin(theta_y), 0],
          [0, 1, 0, 0],
          [-np.sin(theta_y), 0, np.cos(theta_y), 0],
          [0, 0, 0, 1]]

    Rz = [[np.cos(theta_z), -np.sin(theta_z), 0, 0],
          [np.sin(theta_z), np.cos(theta_z), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]

    # Translation matrix
    T = [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]]

    # Final transformation matrix
    transformation = np.dot(T, np.dot(Rz, np.dot(Ry, Rx)))

    return transformation.tolist()

def inverse_transformation_matrix(T):
    """
    Compute the inverse of a 4x4 transformation matrix [R|t] = [R^T | -R^T t].

    Parameters:
        T: 4x4 numpy array, transformation matrix

    Returns:
        inverse_T: 4x4 numpy array, inverse transformation matrix
    """
    # Extract rotation matrix R and translation vector t from T
    R = T[:3, :3]
    t = T[:3, 3]

    # Compute transpose of rotation matrix
    R_transpose = R.T

    # Compute inverse translation
    t_inverse = -np.dot(R_transpose, t)

    # Construct the inverse transformation matrix
    inverse_T = np.hstack((R_transpose, t_inverse.reshape(-1, 1)))
    inverse_T = np.vstack((inverse_T, np.array([0, 0, 0, 1])))

    return inverse_T

def find_cam2base(camera_matrix, base_position_matrix):
    # Find the inverse of the camera matrix
    camera_matrix_inv = inverse_transformation_matrix(camera_matrix)
    print(camera_matrix_inv)
    # Calculate cam2base
    cam2base = np.matmul(camera_matrix_inv, base_position_matrix)

    return cam2base

def intrinsic_matrix(fx, fy, cx, cy):
    """
    Create the camera intrinsic matrix.
    
    Args:
        fx (float): Focal length along the x-axis (in pixels).
        fy (float): Focal length along the y-axis (in pixels).
        cx (float): Principal point x-coordinate (in pixels).
        cy (float): Principal point y-coordinate (in pixels).
        
    Returns:
        list of lists: The camera intrinsic matrix.
    """
    return [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]

def main(args: argparse.Namespace):
    save_folder_name = (
        args.scene_name
        if args.save_suffix is None
        else args.scene_name + "_" + args.save_suffix
    )
    save_root = args.dataset_root + "/" + save_folder_name + "/"
    os.makedirs(save_root, exist_ok=True)

    args.save_folder_name = save_folder_name
    args.save_root = save_root

    # Initialize the controller
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        # scene=get_scene(args.scene_name),
        scene="FloorPlan217",
        # step sizes
        gridSize=args.grid_size,
        snapToGrid=False,
        rotateStepDegrees=30,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        # camera properties
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,
    )


    # event = controller.step("MoveBack")
    # bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('0.jpg', bgr_frame)
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(event.metadata["agent"]["rotation"])
    # event = controller.step(
    #     action="RotateLeft",
    #     degrees=event.metadata["agent"]["rotation"]["y"]
    # )
    # print(event.metadata["agent"]["rotation"])


    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(event.metadata["agent"]["rotation"])
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(angle_deg)

    # event = controller.step(
    #     action="RotateRight",
    #     degrees=angle_deg
    # )
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(event.metadata["agent"]["rotation"])
    # controller.step("MoveBack"
    # )

    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # desired_object = None
    # for obj in event.metadata["objects"]:
    #     if obj.get("objectId") == obj_id:
    #         desired_object = obj
    #         break

    # if desired_object:
    #     # desired_object now holds the dictionary with "objectId" equal to "Fridge"
    #     print("Found the object:", desired_object)

    # for i in range(12):

    #     event = controller.step(
    #         action="RotateRight",
    #         degrees=i*30
    #     )

    #     query = controller.step(
    #         action="GetObjectInFrame",
    #         x=0.50,
    #         y=0.50,
    #         checkVisible=False
    #     )

    #     print(query.metadata["actionReturn"])

    #     bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)

    #     cv2.imwrite(f'{i}.jpg', bgr_frame)
    # save_frame(controller,"0.1")
    # controller.step(
    # action="MoveAhead",
    # moveMagnitude=None)
    # save_frame(controller,"0.2")

    focal_length = 0.5 * args.width * math.tan((args.fov/2)*math.pi / 180)

    # camera intrinsics
    camera_intrinsic = intrinsic_matrix(focal_length, focal_length, args.width/2, args.height/2)

    #Create OG SG
    controller.step("MoveBack")
    event = controller.step("MoveAhead")
    scene_graph = create_scene_graph(event.metadata["objects"])

    # Save scene graph to a file
    file_path = "scene_graph.json"
    with open(file_path, "w") as json_file:
        json.dump(scene_graph, json_file, indent=2)

    print(f"Scene graph saved to {file_path}")


    #Language Query to Decide Planning (LLM Planner) 
    #Eg: Pick the Tomato from the sink and place it in the Fridge

    target_receptacle = "CoffeeTable"
    source_receptacle = "None"
    object = "CoffeeTable"


    controller.step(
            action="AddThirdPartyCamera",
            position={"x": -1, "y": 1, "z": -1},
            rotation={"x": 0, "y": 0, "z": 0},
            fieldOfView=90
        )
    controller.step(
            action="AddThirdPartyCamera",
            position={"x": -1, "y": 1, "z": -1},
            rotation={"x": 0, "y": 0, "z": 0},
            fieldOfView=90
        )
    controller.step(
            action="AddThirdPartyCamera",
            position={"x": -1, "y": 1, "z": -1},
            rotation={"x": 0, "y": 0, "z": 0},
            fieldOfView=90
        )
    controller.step(
            action="AddThirdPartyCamera",
            position={"x": -1, "y": 1, "z": -1},
            rotation={"x": 0, "y": 0, "z": 0},
            fieldOfView=90
        )
    
    # # Define positions and rotations for the three viewpoints
    # viewpoints = [
    #     {"position": {"x": -1.25, "y": 1, "z": -1}, "rotation": {"x": 0, "y": 0, "z": 0}},
    #     {"position": {"x": 1.5, "y": 0.5, "z": -2}, "rotation": {"x": 90, "y": 45, "z": 0}},
    #     {"position": {"x": -0.5, "y": 1.5, "z": -1.5}, "rotation": {"x": 60, "y": -30, "z": 0}}
    # ]

    # # Loop through each viewpoint
    # for i, viewpoint in enumerate(viewpoints):
    #     event = controller.step(
    #         action="UpdateThirdPartyCamera",
    #         thirdPartyCameraId=i,  
    #         position=viewpoint["position"],
    #         rotation=viewpoint["rotation"],
    #         fieldOfView=90
    #     )
        
    # save_images_diff_views(event, prefix="1")

            


###############################################################################################
    #Navigate + Tune Location (To View Object + Effective Manip)
    angle_deg, closest, target_recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
    print(closest)
    event = controller.step(action="Teleport", **closest)
    print(event.metadata["agent"]["rotation"])  
    angle = rotate_angle(controller, target_receptacle)
    # # Rewind the rotation
    event = controller.step(
        action="RotateRight",  # Rewind the rotation by rotating right
        degrees=angle
    )

    # event = controller.step("MoveBack")

    save_frame(controller,"1")
    pos = event.metadata["agent"]["position"]
    rot = event.metadata["agent"]["rotation"]
    # cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
    # cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.5, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
    # cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.5, y=0, z=0.1, theta_x=5, theta_y=10, theta_z=5)

    cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0, z=0, theta_x=0, theta_y=0, theta_z=0)
    cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=0, y=1, z=0, theta_x=0, theta_y=0, theta_z=0)
    cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=1, y=0, z=0, theta_x=0, theta_y=0, theta_z=0)
    cam4_pos,cam4_rot = update_position_rotation(pos,rot,x=0, y=0, z=1, theta_x=0, theta_y=0, theta_z=0)

    viewpoints = [
        {"position": cam1_pos, "rotation": cam1_rot},
        {"position": cam2_pos, "rotation": cam2_rot},
        {"position": cam3_pos, "rotation": cam3_rot},
        {"position": cam4_pos, "rotation": cam4_rot}
    ]
    print(cam1_pos,cam1_rot,cam2_pos,cam2_rot)
    # viewpoints = [
    #     {"position":  event.metadata["agent"]["position"], "rotation": event.metadata["agent"]["rotation"]},
    #     {"position": {"x": 1.5, "y": 0.5, "z": -2}, "rotation": {"x": 90, "y": 45, "z": 0}},
    #     {"position": {"x": -0.5, "y": 1.5, "z": -1.5}, "rotation": {"x": 60, "y": -30, "z": 0}}
    # ]
    cam_2_base_array = []
    # Loop through each viewpoint
    for i, viewpoint in enumerate(viewpoints):
        event = controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=i,  
            position=viewpoint["position"],
            rotation=viewpoint["rotation"],
            fieldOfView=90
        )
        cam_2_base_array.append(transformation_matrix(dict_difference(viewpoint["position"],pos),
                                                      dict_difference(viewpoint["rotation"],rot))) 
       

    
    save_frames_as_dict(event,camera_intrinsic, cam_2_base_array, "1")
    save_images_diff_views(event, prefix="1")








#     # #TargetReceptacle Manipulation (Open)
#     event = controller.step(
#         action="OpenObject",
#         objectId=target_recept_id,
#         openness=1,
#         forceAction=False
#     )

#     save_frame(controller,"2")


#     pos = event.metadata["agent"]["position"]
#     rot = event.metadata["agent"]["rotation"]
#     # cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
#     # cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.5, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
#     # cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.5, y=0, z=0.1, theta_x=5, theta_y=10, theta_z=5)

#     cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=-0.2, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
#     cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.4, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
#     cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.4, y=0.5, z=0.2, theta_x=5, theta_y=7, theta_z=-10)
#     cam4_pos,cam4_rot = update_position_rotation(pos,rot,x=-0.5, y=0.2, z=0.1, theta_x=5, theta_y=10, theta_z=5)

#     viewpoints = [
#         {"position": cam1_pos, "rotation": cam1_rot},
#         {"position": cam2_pos, "rotation": cam2_rot},
#         {"position": cam3_pos, "rotation": cam3_rot},
#         {"position": cam4_pos, "rotation": cam4_rot}
#     ]
#     print(cam1_pos,cam1_rot,cam2_pos,cam2_rot)
#     # viewpoints = [
#     #     {"position":  event.metadata["agent"]["position"], "rotation": event.metadata["agent"]["rotation"]},
#     #     {"position": {"x": 1.5, "y": 0.5, "z": -2}, "rotation": {"x": 90, "y": 45, "z": 0}},
#     #     {"position": {"x": -0.5, "y": 1.5, "z": -1.5}, "rotation": {"x": 60, "y": -30, "z": 0}}
#     # ]
#     cam_2_base_array = []
#     # Loop through each viewpoint
#     for i, viewpoint in enumerate(viewpoints):
#         event = controller.step(
#             action="UpdateThirdPartyCamera",
#             thirdPartyCameraId=i,  
#             position=viewpoint["position"],
#             rotation=viewpoint["rotation"],
#             fieldOfView=90
#         )
#         cam_2_base_array.append(transformation_matrix(dict_difference(viewpoint["position"],pos),
#                                                       dict_difference(viewpoint["rotation"],rot))) 
       

    
#     save_frames_as_dict(event,camera_intrinsic, cam_2_base_array, "2")
        
#     save_images_diff_views(event, prefix="2")


#     #Verify Action + Update SG

#     # #Update SG
#     # action = "Open"
#     # scene_graph = update_scene_graph(scene_graph,action,target_recept_id,None)


# ##########################################################################################
#     #
#     #SourceReceptaple Manipulation (Open)
#     #
# ########################################################################################3

#     #Navigate to Object
#     angle_deg, closest, obj_id = get_angle_and_closest_position(controller,object,scene_graph)

#     controller.step(action="Teleport", **closest)

#     save_frame(controller,"test1")
#     closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
#     angle = rotate_angle(controller, object)
#     controller.step(
#         action="RotateRight",
#         degrees=angle
#     )

#     # controller.step(
#     #     action="MoveAhead"
#     # )

#     bgr_frame = save_frame(controller,"3")
#     # caption, text_prompt = tagging_module(bgr_frame)

#     #Verify
#     # print(text_prompt)
#     # print(closest_items)

#     #Object Pickup
#     event = controller.step(
#     action="PickupObject",
#     objectId=obj_id,
#     forceAction=False,
#     manualInteract=False
#     )

#     bgr_frame = save_frame(controller,"4")
#     # black_image = get_mask_with_pointprompt(bgr_frame)
#     # frame = cv2.cvtColor(black_image,cv2.COLOR_RGB2BGR)


#     #Verify 
#     # caption, text_prompt = tagging_module(frame)
#     # print(caption)
#     # print(text_prompt)

#     #Update SG
#     # action = "Pickup"
#     # scene_graph = update_scene_graph(scene_graph,action,obj_id,None)
#     # print(scene_graph[obj_id])

#     # print(event.metadata["agent"]["position"])



# ############################################################################
#     #Receptacle Navigation

#     angle_deg, closest, recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
#     print(closest)
#     event = controller.step(action="Teleport", **closest)
#     print(event.metadata["agent"]["position"])
#     # event = controller.step("MoveBack")
#     # event = controller.step("MoveRight")
#     save_frame(controller,"test1")
    
#     angle = rotate_angle(controller, target_receptacle)
#     # Rewind the rotation
#     print(angle)
#     controller.step(
#         action="RotateRight",  # Rewind the rotation by rotating right
#         degrees=angle
#     )
#     save_frame(controller,"test2")
#     # event = controller.step("MoveBack")

#     save_frame(controller,"5")
#     # controller.step("MoveBack")

#     #Object Putdown
#     event = controller.step(
#     action="PutObject",
#     objectId=recept_id,
#     forceAction=False,
#     placeStationary=True
# )

#     pos = event.metadata["agent"]["position"]
#     rot = event.metadata["agent"]["rotation"]
#     # cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
#     # cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.5, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
#     # cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.5, y=0, z=0.1, theta_x=5, theta_y=10, theta_z=5)

#     cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=-0.2, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
#     cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.4, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
#     cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.4, y=0.5, z=0.2, theta_x=5, theta_y=7, theta_z=-10)
#     cam4_pos,cam4_rot = update_position_rotation(pos,rot,x=-0.5, y=0.2, z=0.1, theta_x=5, theta_y=10, theta_z=5)

#     viewpoints = [
#         {"position": cam1_pos, "rotation": cam1_rot},
#         {"position": cam2_pos, "rotation": cam2_rot},
#         {"position": cam3_pos, "rotation": cam3_rot},
#         {"position": cam4_pos, "rotation": cam4_rot}
#     ]
#     print(cam1_pos,cam1_rot,cam2_pos,cam2_rot)
#     # viewpoints = [
#     #     {"position":  event.metadata["agent"]["position"], "rotation": event.metadata["agent"]["rotation"]},
#     #     {"position": {"x": 1.5, "y": 0.5, "z": -2}, "rotation": {"x": 90, "y": 45, "z": 0}},
#     #     {"position": {"x": -0.5, "y": 1.5, "z": -1.5}, "rotation": {"x": 60, "y": -30, "z": 0}}
#     # ]

#     cam_2_base_array = []
#     # Loop through each viewpoint
#     for i, viewpoint in enumerate(viewpoints):
#         event = controller.step(
#             action="UpdateThirdPartyCamera",
#             thirdPartyCameraId=i,  
#             position=viewpoint["position"],
#             rotation=viewpoint["rotation"],
#             fieldOfView=90
#         )
#         cam_2_base_array.append(transformation_matrix(dict_difference(viewpoint["position"],pos),
#                                                       dict_difference(viewpoint["rotation"],rot))) 
       

    
#     save_frames_as_dict(event,camera_intrinsic, cam_2_base_array, "3")
        
#     save_images_diff_views(event, prefix="3")

#     save_frame(controller,"6")
#     print(event.metadata["agent"])

#     #Verify
#     # action = "Putdown"
#     # scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    
#     # print(scene_graph[obj_id])
# ###########################################################################3
#     #
#     #TargetReceptacle Manipulation (Close)

#     event = controller.step(
#     action="CloseObject",
#     objectId=recept_id,
#     forceAction=False
#     )
#     pos = event.metadata["agent"]["position"]
#     rot = event.metadata["agent"]["rotation"]
#     # cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)
#     # cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=-0.5, y=0.5, z=-0.2, theta_x=-5, theta_y=-5, theta_z=-10)
#     # cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.5, y=0, z=0.1, theta_x=5, theta_y=10, theta_z=5)

#     cam1_pos,cam1_rot = update_position_rotation(pos,rot,x=0, y=0.3, z=0.5, theta_x=-10, theta_y=6, theta_z=0)
#     cam2_pos,cam2_rot = update_position_rotation(pos,rot,x=0, y=1, z=0, theta_x=-5, theta_y=0, theta_z=-15)
#     cam3_pos,cam3_rot = update_position_rotation(pos,rot,x=-0.5, y=0.5, z=0, theta_x=0, theta_y=-5, theta_z=-10)
#     cam4_pos,cam4_rot = update_position_rotation(pos,rot,x=0, y=0.5, z=0, theta_x=0, theta_y=0, theta_z=-10)

#     viewpoints = [
#         {"position": cam1_pos, "rotation": cam1_rot},
#         {"position": cam2_pos, "rotation": cam2_rot},
#         {"position": cam3_pos, "rotation": cam3_rot},
#         {"position": cam4_pos, "rotation": cam4_rot}
#     ]
#     print(cam1_pos,cam1_rot,cam2_pos,cam2_rot)
#     # viewpoints = [
#     #     {"position":  event.metadata["agent"]["position"], "rotation": event.metadata["agent"]["rotation"]},
#     #     {"position": {"x": 1.5, "y": 0.5, "z": -2}, "rotation": {"x": 90, "y": 45, "z": 0}},
#     #     {"position": {"x": -0.5, "y": 1.5, "z": -1.5}, "rotation": {"x": 60, "y": -30, "z": 0}}
#     # ]
#     cam_2_base_array = []
#     # Loop through each viewpoint
#     for i, viewpoint in enumerate(viewpoints):
#         event = controller.step(
#             action="UpdateThirdPartyCamera",
#             thirdPartyCameraId=i,  
#             position=viewpoint["position"],
#             rotation=viewpoint["rotation"],
#             fieldOfView=90
#         )
#         cam_2_base_array.append(transformation_matrix(dict_difference(viewpoint["position"],pos),
#                                                       dict_difference(viewpoint["rotation"],rot))) 
       

    
#     save_frames_as_dict(event,camera_intrinsic, cam_2_base_array, "4")
        
#     save_images_diff_views(event, prefix="4")

#     save_frame(controller,"7")
    
#     # #Verify

#     # # #Update SG
#     # action = "Close"
#     # scene_graph = update_scene_graph(scene_graph,action,recept_id,None)
# ############################################################################3

#     #
#     #SourceReceptaple Manipulation (Close)
#     #

# ##########################################################################



#     # with open(file_path, "w") as json_file:
#     #     json.dump(scene_graph, json_file, indent=2)

#     # print(f"Scene graph saved to {file_path}")



    print("#########################################################")
    print("Task Completed")
    print("##########################################################")









#     closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)

#     item_names = [item.split('|')[0] for item, _ in closest_items]
#     print(item_names)



#     print("########################")



#     caption, text_prompt = tagging_module(bgr_frame)



    
#     # Add "other item" to capture objects not in the tag2text captions. 
#     # Remove "xxx room", otherwise it will simply include the entire image
#     # Also hide "wall" and "floor" for now...
#     add_classes = ["spoon"]
#     # remove_classes = [
#     #     "room", "kitchen", "office", "house", "home", "building", "corner",
#     #     "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
#     #     "apartment", "image", "city", "blue", "skylight", "hallway", 
#     #     "bureau", "modern", "salon", "doorway", "wall lamp"
#     # ]
#     remove_classes = []
#     bg_classes = ["wall", "floor", "ceiling"]

#     classes = process_tag_classes(
#     text_prompt,
#     add_classes = add_classes,
#     remove_classes = remove_classes,
# )

#     grounding_dino_model = Model(
#         model_config_path=GROUNDING_DINO_CONFIG_PATH, 
#         model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
#         device=args.device
#     )

#     box_threshold=0.2
#     text_threshold=0.2
#     nms_threshold=0.5

#     detections = grounding_dino_model.predict_with_classes(
#         image=bgr_frame, # This function expects a BGR image...
#         classes=classes,
#         box_threshold=box_threshold,
#         text_threshold=text_threshold,
#     )
            


#     if len(detections.class_id) > 0:
#         ### Non-maximum suppression ###
#         # print(f"Before NMS: {len(detections.xyxy)} boxes")
#         nms_idx = torchvision.ops.nms(
#             torch.from_numpy(detections.xyxy), 
#             torch.from_numpy(detections.confidence), 
#             nms_threshold
#         ).numpy().tolist()
#         # print(f"After NMS: {len(detections.xyxy)} boxes")

#         detections.xyxy = detections.xyxy[nms_idx]
#         detections.confidence = detections.confidence[nms_idx]
#         detections.class_id = detections.class_id[nms_idx]
        
#         # Somehow some detections will have class_id=-1, remove them
#         valid_idx = detections.class_id != -1
#         detections.xyxy = detections.xyxy[valid_idx]
#         detections.confidence = detections.confidence[valid_idx]
#         detections.class_id = detections.class_id[valid_idx]
        

#         sam_variant = "sam"
#         sam_predictor = get_sam_predictor(sam_variant, args.device)

#         ### Segment Anything ###
#         detections.mask = get_sam_segmentation_from_xyxy(
#             sam_predictor=sam_predictor,
#             image= cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
#             xyxy=detections.xyxy
#         )

#     print(detections)
#     print("###########################################")
#     print(detections.xyxy)
#     print("###########################################")
#     print(detections.confidence)
#     print("###########################################")
#     print(detections.class_id)
#     print("###########################################")
#     print(detections.mask)
#     print("###########################################")
#     clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#         "ViT-H-14", "laion2b_s32b_b79k"
#     )
#     clip_model = clip_model.to(args.device)
#     clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

#                 # Compute and save the clip features of detections  
#     image_crops, image_feats, text_feats = compute_clip_features(
#      cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            
#     ### Visualize results ###
#     annotated_image, labels = vis_result_fast(bgr_frame, detections, classes)
    
#     Image.fromarray(annotated_image).save("Annotated1.jpg")
#     annotated_image_caption = vis_result_slow_caption(
#          cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), detections.mask, detections.xyxy, labels, caption, text_prompt)
#     Image.fromarray(annotated_image_caption).save("Annotated2.jpg")





    

#     closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)

#     item_names = [item.split('|')[0] for item, _ in closest_items]
#     print(item_names)









#     # image1 = cv2.imread('4.jpg')
#     # image2 = cv2.imread('5.jpg')

#     # intersection = cv2.bitwise_and(image1, image2)

#     # # Check if intersection is not empty
#     # if np.all(intersection == 0):
#     #     print("No common intersection found.")
#     # else:
#     #     # Find bounding box of non-zero pixels in intersection
#     #     non_zero_indices = np.argwhere(cv2.cvtColor(intersection, cv2.COLOR_BGR2GRAY) > 0)
#     #     top_left = tuple(np.min(non_zero_indices, axis=0))
#     #     bottom_right = tuple(np.max(non_zero_indices, axis=0))

#     #     # Extract the common intersection patch
#     #     common_patch = intersection[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

#     #     # Create a white background image
#     #     white_background = np.full_like(common_patch, (255, 255, 255), dtype=np.uint8)

#     #     # Paste the common intersection patch onto the white background
#     #     result = white_background.copy()
#     #     result[0:common_patch.shape[0], 0:common_patch.shape[1]] = common_patch

#     #     # Save the resulting image
#     #     cv2.imwrite('result.jpg', result)


#     # output_dir = 'rotation_images/'

#     # # Assuming controller is your AI2-THOR controller object
#     # for _ in range(6):  # 360 degrees / 60 degrees = 6 steps
#     #     controller.step(
#     #         action="RotateRight",
#     #         degrees=60  # Rotate 60 degrees in each step
#     #     )
        
#     #     # Get the frame after rotation
#     #     frame = controller.last_event.frame
        
#     #     # Convert the frame to BGR format (OpenCV uses BGR by default)
#     #     bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
#     #     # Save the frame as an image
#     #     output_image_path = f'frame_{_ + 1}.jpg'  # Naming each frame with a sequential number
#     #     cv2.imwrite(output_image_path, bgr_frame)
        
#     #     print(f"Image saved as {output_image_path}")

#     # print("Rotation completed.")

#     # reachable_file = save_root + "/reachable.json"
#     # with open(reachable_file, "w") as f:
#     #     json.dump(reachable_positions, f)

#     # Get the poses to generate observations
#     if args.sample_method == "random":
#         sampled_poses = sample_pose_random(controller, args.n_sample)
#     elif args.sample_method == "uniform":
#         sampled_poses = sample_pose_uniform(controller, args.n_sample)
#     elif args.sample_method == "from_file":
#         traj_file = save_root + "/" + args.traj_file_name
#         sampled_poses = sample_pose_from_file(traj_file)
#         save_root = os.path.dirname(traj_file)
#     else:
#         raise ValueError("Unknown sample method: {}".format(args.sample_method))

#     # Capture and save the top-down frame
#     top_down_frame, top_down_grid = get_top_down_frame(controller)
#     top_down_path = save_root + "/top_down.png"
#     top_down_frame.save(top_down_path)
#     top_down_path = save_root + "/top_down_grid.png"
#     top_down_grid.save(top_down_path)
    
#     if args.topdown_only:
#         exit(0)

#     # Generate the images according to the trajectory and save them
#     K = compute_intrinsics(args.fov, args.height, args.width)
#     generate_obs_from_poses(
#         controller=controller,
#         K=K,
#         sampled_poses=sampled_poses,
#         save_root=save_root,
#         depth_scale=args.depth_scale,
#     )


def main_interact(args: argparse.Namespace):
    '''
    Interact with the AI2Thor simulator, navigating the robot. 
    The agent trajectory will be saved to a file as a file. 
    Note that this saves the agent pose but not the camera pose. 
    '''
    save_folder_name = (
        args.scene_name + "_interact"
        if args.save_suffix is None
        else args.scene_name + "_" + args.save_suffix
    )
    save_root = args.dataset_root + "/" + save_folder_name + "/"
    os.makedirs(save_root, exist_ok=True)
    
    args.save_folder_name = save_folder_name
    args.save_root = save_root
    
    grid_size = 0.05
    rot_step = 2

    controller = Controller(
        gridSize=grid_size,
        rotateStepDegrees=rot_step,
        snapToGrid=False,
        # scene=get_scene(args.scene_name),
        scene="FloorPlan6",
        # camera properties
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
    )

    load_or_randomize_scene(args, controller)
    
    controller.step(
        action="LookUp",
        degrees=30
    )
    
    agent_logs = controller.interact()

    print("len(agent_logs):", len(agent_logs))

    trajectory_logs = {
        "scene_name": args.scene_name,
        "grid_size": grid_size,
        "rot_step": rot_step,
        "fov": args.fov,
        "height": args.height,
        "width": args.width,
        "agent_logs": agent_logs,
    }

    # Save log into a json file
    if not args.no_save:
        log_path = save_root + "/" + args.traj_file_name
        print("Saving interaction log to: ", log_path)
        with open(log_path, "w") as f:
            json.dump(trajectory_logs, f, indent=2)
            
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Program Arguments")
    parser.add_argument(
        "--dataset_root",
        default=str(Path("~/ldata/ai2thor/").expanduser()),
        help="The root path to the dataset.",
    )
    parser.add_argument(
        "--grid_size",
        default=0.5,
        type=float,
        help="The translational step size in the scene (default 0.25).",
    )
    
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--interact", action="store_true", help="Run in interactive mode. Requires GUI access."
    )
    parser.add_argument(
        "--traj_file_name", type=str, default="trajectory.json", 
        help="The name of the trajectory file to load."
    )
    
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save trajectories from the interaction."
    )
    parser.add_argument(
        "--height", default=480, type=int, help="The height of the image."
    )
    parser.add_argument(
        "--width", default=640, type=int, help="The width of the image."
    )
    parser.add_argument(
        "--fov", default=90, type=int, help="The (vertical) field of view of the camera."
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save the video of the generated RGB frames."
    )
    
    parser.add_argument("--scene_name", default="train_3")
    parser.add_argument("--save_suffix", default=None)
    parser.add_argument("--randomize_lighting", action="store_true")
    parser.add_argument("--randomize_material", action="store_true")

    # Randomly remove objects in the scene
    parser.add_argument(
        "--randomize_remove_ratio",
        default=0.0,
        type=float,
        help="The probability to remove any object in the scene (0.0 - 1.0)",
    )
    parser.add_argument(
        "--randomize_remove_level", 
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="""What kind of objects to remove duing randomization 
        1: all objects except those in NOT_TO_REMOVE; 
        2: objects that are pickupable or moveable;
        3: objects that are pickupable""",
    )
    
    # Randomly moving objects in the scene
    parser.add_argument(
        "--randomize_move_pickupable_ratio",
        default=0.0,
        type=float,
        help="The ratio of pickupable objects to move.",
    )
    parser.add_argument(
        "--randomize_move_moveable_ratio",
        default=0.0,
        type=float,
        help="The ratio of moveable objects to move.",
    )
    
    parser.add_argument(
        "--topdown_only", action="store_true", help="Generate and save only the topdown view."
    )
    parser.add_argument(
        "--depth_scale", default=1000.0, type=float, help="The scale of the depth."
    )
    parser.add_argument(
        "--n_sample",
        default=-1,
        type=int,
        help="The number of images to generate. (-1 means all reachable positions are sampled)",
    )
    parser.add_argument(
        "--sample_method",
        default="uniform",
        choices=["random", "uniform", "from_file"],
        help="The method to sample the poses (random, uniform, from_file).",
    )
    parser.add_argument("--seed", default=0, type=int)
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.interact:
        main_interact(args)
    else:
        main(args)

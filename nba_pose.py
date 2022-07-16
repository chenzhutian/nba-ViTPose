from os import path as osp
import cv2
import json
import argparse
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

def make_parser():
    parser = argparse.ArgumentParser("ViTPose Demo!")
    parser.add_argument("--path", default="", help="path to images folder")

    # Gt bbox
    parser.add_argument("-g", "--gt_bbox", default=None, type=str, help="provide the GT bboxes")
    parser.add_argument("--out", default=None, type=str, help="the root folder to output results")

    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    img_folder = args.path
    game_id = args.path.split('/')[-1]
    save_folder = osp.join(args.out, game_id)

    pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py'
    pose_checkpoint = '/datadrive/ViTPose/pretrained/vitpose-h-multi-coco.pth'
    
    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    keypoint_info = pose_model.cfg.data.test.dataset_info.keypoint_info

    annot_file = args.gt_bbox
    bboxes_by_frame = {}
    with open(annot_file) as f:
        lines = f.readlines()
        for line in lines:
            frame_idx, tid, x, y, w, h, score, _, _, _= line.split(',')
            if frame_idx not in bboxes_by_frame:
                bboxes_by_frame[frame_idx] = []
            bboxes_by_frame[frame_idx].append(({
                'track_id': int(tid), 
                'bbox': (float(x), float(y), float(w), float(h))
            }))
            
    cap = cv2.VideoCapture(f'{img_folder}/../videos/{game_id}.mp4')
    output_video = cv2.VideoWriter(
        filename=f'{save_folder}/{game_id}-keypoint-vis.mp4',
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        isColor=True
    )
    bboxes_by_frame = sorted(bboxes_by_frame.items(), key=lambda x: int(x[0]))
    data_by_frame = {}
    for frame_idx, bboxes in bboxes_by_frame:
        img = cv2.imread(f'{img_folder}/{frame_idx}.jpg')
        pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                               img,
                                                               bboxes,
                                                            #    bbox_thr=0.3,
                                                               format='xywh',
                                                               dataset=pose_model.cfg.data.test.type)
        vis_result = vis_pose_result(pose_model,
                            img,
                            pose_results,
                            dataset=pose_model.cfg.data.test.type,
                            show=False)

        data = []
        for pose in pose_results:
            keypoints = [(keypoint_info[kId]['name'], float(x), float(y), float(s)) for kId, (x, y, s) in enumerate(pose['keypoints'])]
            t, l, b, r = pose['bbox']
            bbox = list(map(float, [t, l, b - t, r - l]))
            # print(bbox, keypoints)
            data.append({'track_id': pose['track_id'], 'keypoints': keypoints, 'bbox': bbox})
        data_by_frame[frame_idx] = data
        
        output_video.write(vis_result) 
    output_video.release()
    with open(f"{save_folder}/{game_id}-keypoint-bbox.json", 'w') as f:
        json.dump(data_by_frame, f)
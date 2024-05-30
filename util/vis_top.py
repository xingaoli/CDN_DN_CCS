import sys
# 
sys.path.insert(0, "/home/xian/Documents/code/DOQ")
import cv2
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T
import argparse
import util.misc as utils
from util.vis_utils import Visualizer

import numpy as np

torch.set_grad_enabled(False)
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_false',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--human_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='hico')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', default="data/hico_20160224_det", type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_name', default='')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--model_name', default='baseline')
    parser.add_argument('--hard_negative', action='store_true')
    parser.add_argument('--ts_begin', default=3, type=int)
    return parser


# COCO classes
 CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
            'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
            'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted_plant', 'bed', 'dining_table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier',
            'toothbrush']
import json

 verb_name = json.load(open("/home/xian/Documents/code/qpic/data/hico_20160224_det/annotations/verb_list.json", "r"))
 verb_name = [item["name"] for item in verb_name]
 assert len(verb_name) == 117


# verb_name = ['smoke',  'call', 'play(cellphone)', 'eat', 'drink',
            'ride',  'hold',  'kick',  'read', 'play (computer)']
# CLASSES = ['person', 'cellphone', 'cigarette',  'drink',  'food',
            'bike',  'motorbike', 'horse',  'ball', 'computer',  'document']
    
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def plot_hoi_results(o_image_path, prob, humanb_boxes, boxes, action_prob, action_index, image_path=None):
    index = 0
    for p, (xmin1, ymin1, xmax1, ymax1), (xmin, ymin, xmax, ymax), ap, ai in \
            zip(prob, humanb_boxes.tolist(), boxes.tolist(), action_prob, action_index):
        img = cv2.imread(o_image_path)
        cv2.rectangle(img, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)),
                      color=(0, 0, 255), thickness=3)
        cv2.rectangle(img, pt1=(int(xmin1), int(ymin1)), pt2=(int(xmax1), int(ymax1)),
                      color=(255, 0, 0), thickness=3)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        text1 = f'{verb_name[int(ai)]}: {ap:0.2f}'
        # img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None
        cv2.putText(img, text, (int(xmin), int(ymin) - 5), 2, 0.6, (255, 0, 0), 1)
        cv2.putText(img, text1, (int(xmin1), int(ymin1) - 5), 2, 0.6, (255, 0, 0), 1)

        if image_path:
            image_path_ = image_path[0:-4] + f"_{index}" + image_path[-4:]
            cv2.imwrite(image_path_, img)
            index += 1


class AttentionVisualizer:
    def __init__(self, model, image_path, device, out_img_path, decoder_idx):
        self.model = model
        self.o_image_path = image_path

        self.im = Image.open(image_path).convert('RGB')
        self.img = transform(self.im).unsqueeze(0).to(device)

        self.conv_features = None
        self.enc_attn_weights = None
        self.dec_attn_weights = None
        self.outputs = None
        self.out_img_path_decode = os.path.join(out_img_path, image_path.split("/")[-1])

        self.has_data = False
        self.decoder_idx = decoder_idx

        import numpy as np
        # todo
        coor = np.load("/home/xian/Documents/code/DOQ/data/hoia/annotations/corre_hoia.npy")
        s1, s2 = coor.shape
        self.coor = np.zeros((s2, s1))
        for i in range(s1): 
            for j in range(s2): 
                if coor[i][j] : self.coor[j][i]=1

        self.compute_features()
        
        # todo
        hoi_list = json.load(open("/home/xian/Documents/code/DOQ/data/hoia/annotations/hoi_list.json", 'r'))
        # hoi_id_to_num = json.load(
        #     open("/home/xian/Documents/code/Test/Dataset/hico_det/annotations/hoi_id_to_num.json", "r"))
        self.name_to_hoi_index = {}
        for item in hoi_list:
            obj = item["object"]
            verb = item["verb"]
            self.name_to_hoi_index[obj + verb] = int(item["id"]) - 1
        # self.hoi_index_to_rare = {int(k) - 1: v["rare"] for k, v in hoi_id_to_num.items()}

    def compute_features(self):
        model = self.model
        # use lists to store the outputs via up-values
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[self.decoder_idx].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        # propagate through the model
        self.outputs = model(self.img)
        # dict_keys(['pred_obj_logits', 'pred_verb_logits', 'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs'])
        probas = self.outputs['pred_obj_logits'].softmax(-1)[0, :, :-1]
        pred_actions = self.outputs['pred_verb_logits'].sigmoid().squeeze(0)
        obj_score, obj_index = torch.max(probas, -1)
        coor_ = self.coor[obj_index.cpu().numpy()]
        pred_actions = pred_actions * torch.from_numpy(coor_).to(pred_actions.device)
        verb_num = 117

        obj_score = obj_score.repeat_interleave(verb_num, dim=0)
        obj_index = obj_index.repeat_interleave(verb_num, dim=0)

        pred_actions = pred_actions.view(-1)
        pred_actions_index = torch.range(0, verb_num-1)
        repeat_num  = 100
        pred_actions_index = pred_actions_index.repeat(repeat_num)
        pred_score = obj_score * pred_actions
        pred_obj_boxes = rescale_bboxes(self.outputs['pred_obj_boxes'][0], self.im.size)
        pred_sub_boxes = rescale_bboxes(self.outputs['pred_sub_boxes'][0], self.im.size)
        pred_obj_boxes = pred_obj_boxes.repeat_interleave(verb_num, dim=0)
        pred_sub_boxes = pred_sub_boxes.repeat_interleave(verb_num, dim=0)
        
        top_k = 10
        self.pred_score, top_index = torch.topk(pred_score, top_k, sorted=True)
        top_index_ = top_index / verb_num
        self.pred_actions_index = pred_actions_index[top_index]
        self.pred_obj_boxes = pred_obj_boxes[top_index]
        self.pred_sub_boxes = pred_sub_boxes[top_index]
        self.obj_index = obj_index[top_index]

        self.dec_attn_weights = dec_attn_weights[0][0]
        self.dec_attn_weights = self.dec_attn_weights.repeat_interleave(verb_num, dim=0)
        self.dec_attn_weights = self.dec_attn_weights[top_index]
        self.top_index = top_index_.cpu().numpy()

        self.has_data = True
        for hook in hooks:
            hook.remove()
        self.conv_features = conv_features[0]

    def vis_decoder_att(self):
        if not self.has_data: return
        h, w = self.conv_features['0'].tensors.shape[-2:]

        fig, axs = plt.subplots(ncols=len(self.pred_obj_boxes), nrows=2, figsize=(22, 7))
        for obj_idx, ax_i, (xmin, ymin, xmax, ymax), (xmin1, ymin1, xmax1, ymax1), ai, dec_weight, index in \
                zip(self.obj_index, axs.T, self.pred_obj_boxes, self.pred_sub_boxes, self.pred_actions_index,
                    self.dec_attn_weights, self.top_index):
            text = f'{CLASSES[obj_idx]}'
            text1 = f'{verb_name[int(ai)]}'
            ax = ax_i[0]
            ax.imshow(dec_weight.view(h, w).cpu())
            ax.axis('off')
            ax = ax_i[1]
            ax.imshow(self.im)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color='blue', linewidth=3))
            ax.add_patch(plt.Rectangle((xmin1, ymin1), xmax1 - xmin1, ymax1 - ymin1,
                                       fill=False, color='red', linewidth=3))
            ax.axis('off')
            ax.set_title(text + " " + text1 + " " + str(index))
        fig.tight_layout()

        # plt.suptitle(self.rare_name)
        if self.out_img_path_decode:
            plt.savefig(self.out_img_path_decode)

    # def vis_decoder_att_combine(self):
    #     if not self.has_data: return
    #     h, w = self.conv_features['0'].tensors.shape[-2:]
    #
    #     for obj_idx, (xmin, ymin, xmax, ymax), (xmin1, ymin1, xmax1, ymax1), ai, dec_weight, index in \
    #             zip(self.obj_index, self.pred_obj_boxes, self.pred_sub_boxes, self.pred_actions_index,
    #                 self.dec_attn_weights, self.top_index):
    #         image = cv2.imread(self.o_image_path)
    #         h1, w1, _ = image.shape
    #
    #         text = f'{CLASSES[obj_idx]}'
    #         text1 = f'{verb_name[int(ai)]}'
    #
    #         heatmap = dec_weight.view(h, w).cpu().numpy()
    #         heatmap = cv2.resize(heatmap, (w1, h1))
    #         heatmap = heatmap / np.max(heatmap)
    #         heatmap = np.uint8(255 * heatmap)
    #         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #         image = image + heatmap * 0.3
    #         cv2.imwrite(self.out_img_path_decode[0:-4] + f"{index}_{text}" + "_" + f"{text1}" + ".jpg", heatmap)

    def vis_decoder_att_combine(self):
        if not self.has_data: return
        h, w = self.conv_features['0'].tensors.shape[-2:]
        i = 0
        # print(self.obj_index.shape, self.pred_sub_boxes.shape)
        # print(self.pred_obj_boxes.shape, self.pred_actions_index.shape)
        # print(self.dec_attn_weights.shape, self.top_index.shape)
        # torch.Size([10]) torch.Size([10, 4])
        # torch.Size([10, 4]) torch.Size([10])
        # torch.Size([10, 850]) (10,)
        for obj_idx, (xmin, ymin, xmax, ymax), (xmin1, ymin1, xmax1, ymax1), ai, dec_weight, index in \
                zip(self.obj_index, self.pred_sub_boxes, self.pred_obj_boxes, self.pred_actions_index,
                    self.dec_attn_weights, self.top_index):
            image = cv2.imread(self.o_image_path)
            h1, w1, _ = image.shape

            cv2.rectangle(image, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)),
                          color=(0, 0, 255), thickness=2)
            cv2.rectangle(image, pt1=(int(xmin1), int(ymin1)), pt2=(int(xmax1), int(ymax1)),
                          color=(0, 255, 0), thickness=2)

            text = f'{CLASSES[obj_idx]}'
            text1 = f'{verb_name[int(ai)]}'
            heatmap = dec_weight.view(h, w).cpu().numpy()
            heatmap = cv2.resize(heatmap, (w1, h1))
            vis = Visualizer(image)

            heatmap = heatmap / np.max(heatmap)
            alpha = heatmap.copy()
            heatmap[heatmap > 0.2] = 1
            heatmap[heatmap != 1] = 0

            # vis.draw_binary_mask(np.ones_like(heatmap), color=[0.31, 0.31, 0.31])
            vis.draw_binary_mask(heatmap, color=[0, 1, 1], alpha=alpha)
            # vis.draw_binary_mask(heatmap, color=[1, 1, 0], alpha=0.9)
            # vis.draw_binary_mask(heatmap, color=[0, 0.5, 0])
            path = self.out_img_path_decode[0:-4] + f"_{i}_{text}" + "_" + f"{text1}" + ".jpg"
            i += 1
            vis.output.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    device = torch.device(args.device)

    model, _, _ = build_model(args)
    # model.load_state_dict(
    #     torch.load(
    #         "/home/xian/Documents/code/qpic/outputs/hico_abla/checkpoint0016.pth")[
    #         "model"])
    # out_img_path = f"/home/xian/Documents/code/qpic/vis/top10/TS/"

    model.load_state_dict(
        torch.load(
            "/home/xian/Documents/code/DOQ/outputs/hoia/baseline/checkpoint0131.pth")[
            "model"])
    out_img_path = f"/home/xian/Documents/code/DOQ/vis/hoia/baseline"

    model.to(device)
    model.eval()

    if not os.path.exists(out_img_path):
        os.mkdir(out_img_path)
    test_anno = json.load(open("/home/xian/Documents/code/DOQ/data/hoia/annotations/test_2019.json", 'r'))

    hoi_list = json.load(open("/home/xian/Documents/code/DOQ/data/hoia/annotations/hoi_list.json", 'r'))
    # hoi_id_to_num = json.load(
    #     open("/home/xian/Documents/code/Test/Dataset/hico_det/annotations/hoi_id_to_num.json", "r"))

    for item in tqdm(test_anno):
        file_name = item["file_name"]
        hoi_annotation = item["hoi_annotation"]
        has=False
        for hoi in hoi_annotation:
            if hoi["category_id"] == 6 or hoi["category_id"] == 8:
                has=True
                break
        if not has:continue
        
        image_path = f'/home/xian/Documents/code/DOQ/data/hoia/images/test/{file_name}' 
        vis = AttentionVisualizer(model, image_path, device, out_img_path, 5)
        # vis.vis_decoder_att()
        vis.vis_decoder_att_combine()

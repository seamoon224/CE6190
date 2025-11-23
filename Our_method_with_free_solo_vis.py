"""
python Our_method_with_free_solo_vis.py \
    --dataset refcocog \
    --split val \
    --eval-only
"""
import argparse
import clip
import torch
import os

Height, Width = 224, 224

from detectron2.checkpoint import DetectionCheckpointer
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer
import tqdm
from PIL import Image  # 新增：用于可视化

from freesolo.engine.trainer import BaselineTrainer

# hacky way to register
import freesolo.data.datasets.builtin
from freesolo.modeling.solov2 import PseudoSOLOv2

# refer dataset
from data.dataset_refer_bert import ReferDataset
from model.backbone import clip_backbone, CLIPViTFM
from utils import default_argument_parser, setup, Compute_IoU, extract_noun_phrase
from collections import defaultdict


# ================= 可视化相关的辅助函数 =================

def tensor_to_pil(img_tensor):
    """
    将 detectron2 风格的图像 tensor 转成 PIL.Image 用于保存/可视化.
    支持:
        - [C,H,W]
        - [1,C,H,W]
        - [H,W,C]
    """
    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        # 假设 [B,C,H,W]，取第 0 张
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        # [C,H,W] -> [H,W,C]
        img = img.permute(1, 2, 0)

    img_np = img.numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype("uint8")
    else:
        img_np = img_np.astype("uint8")

    return Image.fromarray(img_np)


# def overlay_mask(base_img, mask, color=(255, 0, 0), alpha=0.5):
#     """
#     在 base_img 上叠加一个二值 mask, 返回新的 PIL.Image
#     base_img: PIL.Image
#     mask: torch.Tensor 或 numpy, 2D 或 3D (会取第一通道)
#     color: (R,G,B)
#     alpha: 透明度
#     """
#     if isinstance(mask, torch.Tensor):
#         m = mask.detach().cpu().numpy()
#     else:
#         m = np.array(mask)

#     # 如果 mask 是 [C,H,W]，取第一通道
#     if m.ndim == 3:
#         m = m[0]
#     # 二值化
#     m = (m > 0.5).astype("uint8") * 255

#     mask_img = Image.fromarray(m).convert("L")
#     # 调整到和原图同尺寸
#     if mask_img.size != base_img.size:
#         mask_img = mask_img.resize(base_img.size, resample=Image.NEAREST)

#     color_img = Image.new("RGBA", base_img.size, color + (int(255 * alpha),))
#     base_rgba = base_img.convert("RGBA")

#     # 用 mask 作为 alpha 叠加
#     out = Image.composite(color_img, base_rgba, mask_img)
#     return out.convert("RGB")

def overlay_mask(base_img, mask, color=(255, 0, 0), alpha=0.3):
    """
    在 base_img 上叠加一个二值 mask, 返回新的 PIL.Image
    base_img: PIL.Image
    mask: torch.Tensor 或 numpy, 2D 或 3D (会取第一通道)
    color: (R,G,B)
    alpha: 透明度 (0~1)
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.array(mask)

    # 如果 mask 是 [C,H,W]，取第一通道
    if m.ndim == 3:
        m = m[0]

    # 二值化
    m = (m > 0.5).astype("uint8") * 255
    mask_img = Image.fromarray(m).convert("L")

    base_rgba = base_img.convert("RGBA")

    # 创建“只在 mask 位置有颜色”的 overlay
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    color_img = Image.new("RGBA", base_rgba.size,
                          color + (int(255 * alpha),))  # 带 alpha 的纯色图

    # 用 mask 决定哪里贴上有 alpha 的颜色
    if mask_img.size != base_rgba.size:
        mask_img = mask_img.resize(base_rgba.size, resample=Image.NEAREST)
    overlay.paste(color_img, (0, 0), mask_img)

    # 和原图进行 alpha 合成
    out = Image.alpha_composite(base_rgba, overlay)
    return out.convert("RGB")



# ================= 主逻辑 =================

def main(args, Height, Width):
    assert args.eval_only, 'Only eval_only available!'
    cfg = setup(args)

    if args.dataset == 'refcocog':
        args.splitBy = 'umd'  # umd or google in refcocog
    else:
        args.splitBy = 'unc'  # unc in refcoco, refcoco+,

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ReferDataset(
        args,
        image_transforms=None,
        target_transforms=None,
        split=args.split,
        eval_mode=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=False
    )

    Trainer = BaselineTrainer
    Free_SOLO = Trainer.build_model(cfg)
    Free_SOLO.eval()

    mode = 'ViT'  # or Res
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'

    Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM(model_name='ViT-B/32').to(device)

    DetectionCheckpointer(Free_SOLO, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    nlp = spacy.load('en_core_web_lg')

    cum_I, cum_U = 0, 0
    m_IoU = []

    v = 0.85 if args.dataset == 'refcocog' else 0.95    # v=alpha
    r = 0.5   # r=beta

    # 可视化保存目录
    vis_dir = './vis_results'
    os.makedirs(vis_dir, exist_ok=True)

    tbar = tqdm.tqdm(data_loader)

    for i, data in enumerate(tbar):
        image, target, clip_embedding, sentence_raw = data
        clip_embedding, target = clip_embedding.squeeze(1).to(device), target.to(device)

        # 这里的 image 是 ReferDataset 给 detectron2 的结构, Free_SOLO 接受的就是这个
        pred = Free_SOLO(image)[0]

        pred_masks = pred['instances'].pred_masks
        pred_boxes = pred['instances'].pred_boxes

        if len(pred_masks) == 0:
            print('No pred masks')
            continue

        original_imgs = torch.stack(
            [
                T.Resize((height, width))(img.to(pred_masks.device))
                for img, height, width in zip(
                    image[0]['image'], image[0]['height'], image[0]['width']
                )
            ],
            dim=0
        )  # [1, 3, H_orig, W_orig]
        resized_imgs = torch.stack(
            [
                T.Resize((Height, Width))(img.to(pred_masks.device))
                for img in image[0]['image']
            ],
            dim=0
        )  # [1,3,224,224]

        cropped_imgs = []

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(pred_masks.device)

        for pred_box, pred_mask in zip(pred_boxes.__iter__(), pred_masks):
            pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
            masked_image = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean

            x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
            masked_image = TF.resized_crop(
                masked_image.squeeze(0),
                y1, x1, (y2 - y1), (x2 - x1),
                (Height, Width)
            )
            cropped_imgs.append(masked_image)

        cropped_imgs = torch.stack(cropped_imgs, dim=0)

        if mode == 'Res':
            mask_features = Model.feature_map_masking(resized_imgs, pred_masks)
        else:
            mask_features = Model(
                resized_imgs,
                pred_masks,
                masking_type='token_masking',
                masking_block=9
            )

        if mode == 'Res':
            crop_features = Model.get_gloval_vector(cropped_imgs)
        else:
            crop_features = Model(
                cropped_imgs,
                pred_masks=None,
                masking_type='crop'
            )

        visual_feature = v * mask_features + (1 - v) * crop_features

        # 这里每个样本可能对应多个 sentence_raw（多条表达），逐条处理
        for j, sentence in enumerate(sentence_raw):
            sentence = sentence[0].lower()
            doc = nlp(sentence)
            sentence_for_spacy = []

            for token in doc:
                if token.text == ' ':
                    continue
                sentence_for_spacy.append(token.text)

            sentence_for_spacy = ' '.join(sentence_for_spacy)
            sentence_token = clip.tokenize(sentence_for_spacy).to(device)
            noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(
                sentence_for_spacy, nlp, need_index=True
            )
            noun_phrase_token = clip.tokenize(noun_phrase).to(device)

            if mode == 'Res':
                sentence_features = Model.get_text_feature(sentence_token)
                noun_phrase_features = Model.get_text_feature(noun_phrase_token)
            else:
                sentence_features = Model.model.encode_text(sentence_token)
                noun_phrase_features = Model.model.encode_text(noun_phrase_token)

            text_ensemble = r * sentence_features + (1 - r) * noun_phrase_features

            if mode == 'Res':
                score = Model.calculate_similarity_score(visual_feature, text_ensemble)
            else:
                score = Model.calculate_score(visual_feature, text_ensemble)

            max_index = torch.argmax(score)
            result_seg = pred_masks[max_index]   # 预测 mask

            # -------- 可视化：前 50 个样本，每个样本只保存第一个表达对应的 mask --------
            if i < 50 and j == 0:
                try:
                    # 原图：取 image[0]['image'][0] 这一张
                    # base_img_tensor = image[0]['image'][0]
                    # base_img = tensor_to_pil(base_img_tensor)
                    file_name = image[0]['file_name']

                    # 如果 file_name 是 list → 取第一个
                    if isinstance(file_name, list):
                        file_name = file_name[0]

                    # 用 os.path.join 拼接路径
                    img_path = os.path.join(
                        '/home/haiyue/codes/project/baseline/CE6190/individual/Zero-shot-RIS/refer/data/images/mscoco/images/train2014',
                        file_name
                    )

                    # 保存 sentence_raw
                    sent_txt_path = os.path.join(vis_dir, f'{i:05d}_sentence.txt')
                    with open(sent_txt_path, 'w') as ftxt:
                        ftxt.write(sentence + "\n")


                    print(f"Visualizing image: {img_path}")
                    base_img = Image.open(img_path).convert("RGB")

                    # 保存原图
                    base_img.save(os.path.join(vis_dir, f'{i:05d}_orig.jpg'))

                    # 预测 mask 叠加（红色）
                    pred_overlay = overlay_mask(base_img, result_seg, color=(255, 0, 0), alpha=0.45)
                    pred_overlay.save(os.path.join(vis_dir, f'{i:05d}_pred.jpg'))

                    # GT mask：target 可能是 [1,H,W] 或 [H,W]
                    gt_mask = target
                    if isinstance(gt_mask, torch.Tensor):
                        if gt_mask.ndim == 3:
                            gt_mask = gt_mask[0]
                    gt_overlay = overlay_mask(base_img, gt_mask, color=(255, 0, 0), alpha=0.45)
                    gt_overlay.save(os.path.join(vis_dir, f'{i:05d}_gt.jpg'))
                except Exception as e:
                    print(f"Visualization error at index {i}: {e}")

            # -------- IoU 统计 --------
            _, m_IoU, cum_I, cum_U = Compute_IoU(result_seg, target, cum_I, cum_U, m_IoU)

        # 如果只想保存前 50 张图像的可视化，不影响 IoU 统计，可以继续遍历；
        # 若你希望前 50 张之后直接停止整轮评估，可以在这里加：
        if i >= 49:
            break

    # -------- 结果写入日志 --------
    os.makedirs('./result_log', exist_ok=True)
    f = open('./result_log/our_method_with_free_solo_eval.txt', 'a')
    f.write(f'\n\n CLIP Model: {mode}'
            f'\nDataset: {args.dataset} / {args.split} / {args.splitBy} / alpha={v} / beta={r}'
            f'\nOverall IoU / mean IoU')

    overall = cum_I * 100.0 / cum_U
    mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0

    f.write(f'\n{overall:.2f} / {mean_IoU:.2f}')
    f.close()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # 这行仍然是给 cfg 传 OUTPUT_DIR 和 MODEL.WEIGHTS 的快捷方式
    opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl',
            'MODEL.WEIGHTS', 'checkpoints/FreeSOLO_R101_30k_pl.pth']
    args.opts = opts
    print(args.opts)
    main(args, Height, Width)

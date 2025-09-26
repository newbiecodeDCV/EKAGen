import os
import argparse
import pickle
import numpy as np
import torch
from PIL import Image

from models import caption, utils
from models.model import swin_tiny_patch4_window7_224 as create_model
from datasets import xray
from datasets.tokenizers import Tokenizer
from datasets.utils import nested_tensor_from_tensor_list
from utils.engine import create_caption_and_mask


def build_diagnosisbot(num_classes: int, detector_weight_path: str):
    model = create_model(num_classes=num_classes)
    assert os.path.exists(detector_weight_path), f"file: '{detector_weight_path}' does not exist."
    model.load_state_dict(torch.load(detector_weight_path, map_location=torch.device('cpu')), strict=True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    return model


def run_inference(config):
    """Sinh báo cáo cho 1 ảnh duy nhất và trả về chuỗi báo cáo."""
    device = torch.device(config.device)

    # Load thresholds for DiagnosisBot -> prior
    thresholds = None
    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    # Seed
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build tokenizer early to sync vocab size with checkpoint/dataset
    threshold = 10 if config.dataset_name == 'mimic_cxr' else 3
    tmp_tokenizer = Tokenizer(ann_path=config.anno_path, threshold=threshold, dataset_name=config.dataset_name)
    # Tokenizer uses indices starting at 3; embedding size should cover all indices
    config.vocab_size = max(tmp_tokenizer.idx2token.keys()) + 1 if hasattr(tmp_tokenizer, 'idx2token') else (
        tmp_tokenizer.get_vocab_size() + 3)

    # Models
    detector = build_diagnosisbot(config.num_classes, config.detector_weight_path).to(device)
    model, _ = caption.build_model(config)
    model.to(device)

    # Load caption checkpoint
    if os.path.exists(config.test_path):
        weights_dict = torch.load(config.test_path, map_location='cpu')['model']
        model.load_state_dict(weights_dict, strict=False)
    else:
        raise FileNotFoundError(f"--test_path not found: {config.test_path}")

    if not getattr(config, 'image_path', None):
        raise ValueError("Vui lòng cung cấp --image_path để suy luận 1 ảnh.")

    # Single-image inference
    report = generate_single_report(
        image_path=config.image_path, config=config, device=device,
        model=model, class_model=detector, thresholds=thresholds,
    )
    return report


def _load_single_tensors(image_path: str, config):
    # Build transforms identical to validation
    _, val_transform = xray.get_transform(MAX_DIM=config.image_size)
    class_transform = xray.transform_class

    image = Image.open(image_path).resize((300, 300)).convert('RGB')
    class_image = image.copy()
    com_image = image.copy()

    # Optional ADA boosting using precomputed mask if exists (best effort)
    mask_path = None
    if config.dataset_name == 'mimic_cxr':
        mask_path = image_path.replace('images300', 'images300_array').replace('.jpg', '.npy')
    else:
        mask_path = image_path.replace('images', 'images300_array').replace('.png', '.npy')
    if os.path.exists(mask_path):
        mask_arr = np.load(mask_path)
        if (np.sum(mask_arr) / 90000) > getattr(config, 'theta', 0.0):
            image_arr = np.asarray(image)
            boost_arr = image_arr * np.expand_dims(mask_arr, 2)
            weak_arr = image_arr * np.expand_dims(1 - mask_arr, 2)
            image = Image.fromarray(boost_arr + (weak_arr * getattr(config, 'gamma', 0.4)).astype(np.uint8))

    image = val_transform(image)
    com_image = val_transform(com_image)
    nested = nested_tensor_from_tensor_list(image.unsqueeze(0), max_dim=config.image_size)
    com_nested = nested_tensor_from_tensor_list(com_image.unsqueeze(0), max_dim=config.image_size)

    class_image = class_transform(class_image)

    images = nested.tensors.squeeze(0)
    masks = nested.mask.squeeze(0)
    com_images = com_nested.tensors.squeeze(0)
    com_masks = com_nested.mask.squeeze(0)
    return images, masks, com_images, com_masks, class_image


def generate_single_report(image_path: str, config, device, model, class_model, thresholds):
    # Prepare tokenizer consistent with dataset
    threshold = 10 if config.dataset_name == 'mimic_cxr' else 3
    tokenizer = Tokenizer(ann_path=config.anno_path, threshold=threshold, dataset_name=config.dataset_name)

    images, masks, _, _, class_image = _load_single_tensors(image_path, config)
    samples = utils.NestedTensor(images.unsqueeze(0), masks.unsqueeze(0)).to(device)

    model.eval(); class_model.eval()
    caption, cap_mask = create_caption_and_mask(config.start_token, config.max_position_embeddings, 1)
    for i in range(config.max_position_embeddings - 1):
        logit = class_model(class_image.unsqueeze(0).to(device))
        thresholded_predictions = 1 * (logit.cpu().numpy() > thresholds)
        predictions = model(samples.to(device), caption.to(device), cap_mask.to(device), [thresholded_predictions, tokenizer])
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if i == config.max_position_embeddings - 2:
            break
        caption[:, i + 1] = predicted_id
        cap_mask[:, i + 1] = False

    pred_tokens = [t for t in caption.squeeze(0).cpu().numpy().tolist() if t not in [config.start_token, config.end_token, 0]]
    report = tokenizer.decode(pred_tokens)
    return report


def build_argparser():
    parser = argparse.ArgumentParser(description="EKAGen single-image inference")

    # Basic runtime
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    # Các tham số dưới không ảnh hưởng trực tiếp đến single-image inference, nhưng
    # là bắt buộc khi build model/backbone
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--dilation', type=bool, default=True)

    # Transformer / tokenizer config (keep in sync with training)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=4253)
    parser.add_argument('--start_token', type=int, default=1)
    parser.add_argument('--end_token', type=int, default=2)
    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', type=int, default=True)

    # DiagnosisBot / thresholds
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--thresholds_path', type=str, default='./datasets/thresholds.pkl')
    parser.add_argument('--detector_weight_path', type=str, default='./weight_path/diagnosisbot.pth')
    parser.add_argument('--knowledge_prompt_path', type=str, default='./knowledge_path/knowledge_prompt_iu.pkl')
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    # ADA defaults (match main.py test settings)
    parser.add_argument('--theta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Dataset
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--anno_path', type=str, default='../dataset/mimic_cxr/annotation.json')
    parser.add_argument('--data_dir', type=str, default='../dataset/mimic_cxr/images300')

    # Caption model checkpoint
    parser.add_argument('--test_path', type=str, default='./weight_path/caption_model.pth',
                        help='Path to caption checkpoint .pth')
    # Single image path for one-sample generation
    parser.add_argument('--image_path', type=str, default='', help='Absolute path to image (jpg/png) to infer')

    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    result = run_inference(args)
    if isinstance(result, str):
        # In single-image mode, print only the generated report string
        print(result)



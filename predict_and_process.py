import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    full_img = full_img.convert("RGB")
    img_np = BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    img = torch.from_numpy(img_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)
  
def post_process_image(img_path, mask_path, processed_path):
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    if not os.path.exists(mask_path):
        logging.warning(f"No mask for {img_path}, skipping post-processing.")
        return
    mask_img = Image.open(mask_path).convert('L')
    mask_data = list(mask_img.getdata())
    img = Image.open(img_path).convert('RGB')
    img_data = img.load()
    width, height = img.size
    for i, m in enumerate(mask_data):
        if m != 0:
            x = i % width
            y = i // width
            img_data[x, y] = (255, 255, 255)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    img.convert("L").save(processed_path)
    logging.info(f"Processed image saved to {processed_path}")

def is_image_file(fname):
    return fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))

def main():
    parser = argparse.ArgumentParser(description='Predict masks and post-process images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Model file')
    parser.add_argument('--input', '-i', metavar='INPUT', required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='Output directory for masks (default: output/<input_name>)')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save masks')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize predictions')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Mask threshold for binary')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for input images')
    parser.add_argument('--bilinear', action='store_true', default=True,
                        help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1,
                        help='Number of classes')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    inputs = []
    if os.path.isdir(args.input):
        inputs = [os.path.join(args.input, f) for f in os.listdir(args.input) if is_image_file(f)]
        base_name = os.path.basename(os.path.normpath(args.input))
        mask_output_dir = args.output or os.path.join('output', base_name)
    else:
        inputs = [args.input]
        mask_output_dir = args.output or 'output'

    os.makedirs(mask_output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        processed_dir = os.path.join('processed', base_name)
    else:
        processed_dir = 'processed'
    os.makedirs(processed_dir, exist_ok=True)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model} on {device}')
    net.to(device=device)
    state = torch.load(args.model, map_location=device)
    mask_values = state.pop('mask_values', [0, 1])
    net.load_state_dict(state)
    for img_path in inputs:
        fname = os.path.basename(img_path)
        name, _ = os.path.splitext(fname)
        logging.info(f'Predicting {fname}...')
        img = Image.open(img_path)

        mask = predict_img(net, img, device,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold)
        mask_file = os.path.join(mask_output_dir, f"{name}_OUT.png")
        if not args.no_save:
            mask_img = mask_to_image(mask, mask_values)
            os.makedirs(mask_output_dir, exist_ok=True)
            mask_img.save(mask_file)
            logging.info(f'Mask saved to {mask_file}')

        if args.viz:
            logging.info('Visualizing...')
            plot_img_and_mask(img, mask)
        proc_file = os.path.join(processed_dir, fname)
        post_process_image(img_path, mask_file, proc_file)


if __name__ == '__main__':
    main()

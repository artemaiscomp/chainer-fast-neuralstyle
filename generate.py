from __future__ import print_function
import numpy as np
import argparse
from PIL import Image, ImageFilter
import time

import glob

import chainer
from chainer import cuda, Variable, serializers
from net import *

# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')


def get_model_and_backend(model_file, args_gpu):
    model = FastStyleNet()
    serializers.load_npz(model_file, model)
    
    if args_gpu >= 0:
        cuda.get_device(args_gpu).use()
        model.to_gpu()
    
    xp = np if args_gpu < 0 else cuda.cupy

    return model, xp

def apply_model(original_image, padding):
    #image = np.asarray(original, dtype=np.float32).transpose(2, 0, 1)
    image = np.asarray(original_image).astype(np.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)

    if padding > 0:
        image = np.pad(image, [[0, 0], [0, 0],
                [padding, padding],
                [padding, padding]], 'symmetric')
    
    x = Variable(xp.asarray(image))
    y = model(x)

    result = cuda.to_cpu(y.data)

    if padding > 0:
        result = result[:, :,
                padding:-padding,
                padding:-padding]
    
    return np.uint8(result[0].transpose((1, 2, 0)))


def apply_style_in_image(filename, args, print_time=False):
    original = Image.open(filename).convert('RGB')

    if print_time:
        start = time.time()            
    result = apply_model(original, args.padding)
            
    med = Image.fromarray(result)
    if args.median_filter > 0:
        med = med.filter(ImageFilter.MedianFilter(args.median_filter))
    if args.keep_colors:
        med = original_colors(original, med)
    
    if print_time:
        print(time.time() - start, 'sec')
    
    return med


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
    parser.add_argument('input')
    parser.add_argument('type', choices=('unique', 'folder'))
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='models/style.model', type=str)
    parser.add_argument('--out', '-o', default='out.jpg', type=str)
    parser.add_argument('--median_filter', default=3, type=int)
    parser.add_argument('--padding', default=50, type=int)
    parser.add_argument('--keep_colors', action='store_true')
    parser.set_defaults(keep_colors=False)
    args = parser.parse_args()

    model, xp = get_model_and_backend(args.model, args.gpu)

    if args.type == 'unique':
        med = apply_style_in_image(args.input, args)
        med.save(args.out)
    else:
        fnames = glob.glob(args.input)
        for fname in fnames:
            med = apply_style_in_image(fname, args)
            med.save(args.out + '/' + fname.split('/')[-1])
        

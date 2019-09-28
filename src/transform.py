from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import h5py
import tensorflow as tf

from skimage import color
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from imageio import imwrite

from loss import dummy_loss
import nets
from style_util import StyleUtil
from ImageUtil import *

import time
import os,sys
import argparse
import numpy as np 

# 画像のスタイル変換機能を提供するクラス
class StyleTransferManager:
    # コンストラクタ
    #
    # input
    #   sutil          スタイルユーティリティクラス
    #   original_color 初期値0, 0~1
    #   blend          初期値0.1, 0~1
    #   size_media_filter 初期値3, メディアンフィルタサイズ
    #   size_image        初期値256
    def __init__(self, sutil, original_color=0, blend=0.1,
                 size_media_filter=3, size_image=256):
        self.style_util = sutil
        self.original_color = original_color
        self.blend_alpha    = blend
        self.size_media_filter   = size_media_filter


    # 画像変換処理を行います。
    #
    # input
    #   file_input  入力画像のファイルパス
    #   style       適用するスタイルを表す文字列
    #   file_output 出力先のファイルパス
    #
    def transfer_file(self, file_input, style, file_output):
        # 入力ファイルチェック
        if os.path.isfile(file_input) == False:
            print("File Not Found:",file_input)
            return

        # 出力先ディレクトリチェック
        dir_output = os.path.dirname(file_output)
        if os.path.isdir(dir_output) == False:
            print("Directory Not Found:",dir_output,"file_path:",file_output)
            return

        # スタイルチェック
        style_list = self.style_util.get_style_list()
        if self.style_util.is_valid_style(style) == False:
            print("Invalid style.{0}".format(style))
            print("Valild style list:", style_list)
            return

        img_input = Image.open(file_input)
        img_output = self.transfer_image(img_input, style)
        imwrite(file_output, img_output)


    # 画像変換処理を行います。
    #
    # input
    #   img_input  入力画像のImage データ
    #   style       適用するスタイルを表す文字列
    # output
    #   img_output 変換後のImageデータ
    def transfer_image(self, img_input, style):
        aspect_ratio, x = ImageUtil.reflect(img_input, size_multiple=4)
        img_width= img_height = x.shape[1]

        net = nets.image_transform_net(img_width,img_height)
        model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)
        model.summary()

        # Dummy loss since we are learning from regularizes
        model.compile(Adam(),  dummy_loss)
        model.load_weights(self.style_util.get_style_weight_path(style), by_name=False)

        t1 = time.time()
        y = net.predict(x)[0] 
        y = ImageUtil.crop(y, aspect_ratio)

        print("process: %s" % (time.time() -t1))
        ox = ImageUtil.crop(x[0], aspect_ratio)

        # メディアンフィルタ
        img_output =  self._median_filter_colors(y, self.size_media_filter)

        if self.blend_alpha > 0:
            img_output = self._blend(ox, img_output, self.blend_alpha)

        if self.original_color > 0:
            img_output = self._original_colors(ox, img_output,self.original_color )

        return img_output 


    """
    Applies a median filer to all colour channels
    """
    def _median_filter_colors(self, im_small, window_size):
        ims = []
        for d in range(3):
            im_conv_d = median_filter(im_small[:,:,d], size=(window_size,window_size))
            ims.append(im_conv_d)

        im_conv = np.stack(ims, axis=2).astype("uint8")
        return im_conv

    def _blend(self, original, stylized, alpha):
        return alpha * original + (1 - alpha) * stylized

    # from 6o6o's fork. 
    # https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
    def _original_colors(self, original, stylized,original_color):
        # Histogram normalization in v channel
        ratio=1. - original_color 

        hsv = color.rgb2hsv(original/255)
        hsv_s = color.rgb2hsv(stylized/255)
        hsv_s[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
        img = color.hsv2rgb(hsv_s)    
        return img


def main(args):
    style= args.style
    file_prefix =args.output
    dir_output =args.dir_output
    file_input = args.input
    original_color = args.original_color
    blend_alpha = args.blend
    media_filter = args.media_filter

    sutil = StyleUtil("../conf", dir_output)
    manager = StyleTransferManager(sutil,
                original_color=0, blend=0.1, size_media_filter=3, size_image=256)
    file_output = '%s/%s_%s.png' % (dir_output,file_prefix, style)
    manager.transfer_file(file_input, style, file_output)

    styles = sutil.get_style_list()
    print("style list",styles)
    isvalid = sutil.is_valid_style("mirror")
    print("is valid style:","mirror",isvalid)
    isvalid = sutil.is_valid_style("mirror2")
    print("is valid style:","mirror2",isvalid)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')

    parser.add_argument('--input', '-i', default=None, required=True,type=str,
                        help='input file name')

    parser.add_argument('--output', '-o', default=None, required=True,type=str,
                        help='output file name without extension')
    parser.add_argument('--dir_output', '-d', default="../images/output", required=True,type=str,
                        help='output directory')

    parser.add_argument('--original_color', '-c', default=0, type=float,
                        help='0~1 for original color')

    parser.add_argument('--blend', '-b', default=0, type=float,
                        help='0~1 for blend with original image')
    parser.add_argument('--media_filter', '-f', default=3, type=int,
                        help='media_filter size')
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)


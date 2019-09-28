import sys
import numpy as np
import cv2
from PIL import Image
from imageio import imwrite

from skimage.transform import resize
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

# しきい値 コンフィデンス
CONF_THRESHOLD = 0.5
# しきい値 Non Maximum Suppression
NMS_THRESHOLD = 0.4

IMG_WIDTH = 416
IMG_HEIGHT = 416

# BGR(OpenCV形式) で表した色
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# バウンディングボックスを表現するクラス
class BBox:
    def __init__(self, left, right, top, bottom):
        self.left   = left
        self.right  = right
        self.top    = top
        self.bottom = bottom
        self.width  = self.right - self.left
        self.height = self.bottom - self.top 

    def setSize(self, left, top, width, height):
        self.width  = width
        self.height = height
        self.left   = left
        self.top    = top
        self.right  = self.left + self.width
        self.bottom = self.top  + self.height

# 画像処理ユーティリティクラス
class ImageUtil:

    # 画像サイズの取得
    #
    # input
    #   file_image 画像ファイルのパス
    # output
    #   height,widht,channel数を返します
    @classmethod
    def get_image_size(self, file_image):
        # 対象画像読み込み
        img = cv2.imread(file_image, cv2.IMREAD_COLOR)

        # 画像の大きさを取得
        height, width, channels = img.shape[:3]
        return height,width,channels


    # 画像のリサイズ
    #
    # input
    #   file_in 入力画像ファイルパス
    #   height  リサイズ後の高さ
    #   width   リサイズ後の幅
    #   file_out 出力画像ファイルパス
    @classmethod
    def resize_file(self, file_in, height, width, file_out):
        # 対象画像読み込み
        img = Image.open(file_in)
        img = img.resize((width, height),Image.NEAREST)
        imwrite(file_out, img)


    # OpenCV format からPIL format にイメージデータ変換
    #
    # input
    #   image 変換対象のイメージデータ
    # output
    #   変換後のイメージデータ
    @classmethod
    def opencv2PIL(self, image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)


    # イメージデータの白色化
    #
    # input
    #   image 変換対象のイメージデータ
    # output
    #   変換後のイメージデータ
    @classmethod
    def bgr2gray(self, image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


    @classmethod
    def adjust_rect(self, out_box, img_width, img_height):
        #img_width = 1280
        #img_height = 720 -> detected 704
        top,left,bottom,right = out_box
        top = max(0.0,top)
        left = max(0.0,left)
        width = right - left
        height = bottom - top
        new_width = max(width,height)

        #left = max(0,left -8)
        bottom = min(top + new_width,img_height)
        right = min(left+new_width,img_width)
        return (int(top),int(left),int(bottom),int(right))


    # Draw the predicted bounding box
    @classmethod
    def draw_predict(self, frame, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

        text = '{:.2f}'.format(conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, label_size[1])
        cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)


    @classmethod
    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin
        return left, top, right, bottom


    # Util function to open, resize and format pictures into appropriate tensors
    #
    #    Preprocess the image so that it can be used by Keras.
    #    Args:
    #        image_path: path to the image
    #        img_width: image width after resizing. Optional: defaults to 256
    #        img_height: image height after resizing. Optional: defaults to 256
    #    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
    #             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"
    @classmethod
    def resize_2tensors(self, image_path, img_width=256, img_height=256):
        img = Image.open(image_path)
        img = img.resize((img_width, img_height),Image.NEAREST)
        img_arr = np.asarray(img)
        img_arr = img_arr.astype(np.float32)
        img = np.expand_dims(img_arr, axis=0)
        return img


    # 画像の反転
    # TODO: 実装が複雑なので簡易化できないか調査が必要
    #
    # input
    #   img_input      画像Imageデータ
    #   size_multiple  高さ、幅のサイズが指定数の倍数に補正します
    # output
    #   aspect_ratio  高さ/幅の割合
    #   img_arr       反転結果の画像を表すテンソル
    @classmethod
    def reflect(self, img_input, size_multiple=4):
        # Prevents crashes due to PNG images (ARGB)
        img = img_input
        org_w = img.width
        org_h = img.height

        aspect_ratio = org_h/org_w
        # Make sure width/height is a multiple of 4
        sw = (org_w // size_multiple) * size_multiple
        sh = (org_h // size_multiple) * size_multiple

        size  = sw if sw > sh else sh
        pad_w = (size - sw) // 2
        pad_h = (size - sh) // 2

        tf_session = K.get_session()
        kvar = K.variable(value=np.asarray(img))

        paddings = [[pad_w,pad_w],[pad_h,pad_h],[0,0]]
        squared_img = tf.pad(kvar, paddings, mode='REFLECT', name=None)
        img_arr = K.eval(squared_img)
        img = Image.fromarray(np.uint8(img_arr))

        img = img.resize((size, size),Image.NEAREST)
        img_arr = np.asarray(img)
        img_arr = img_arr.astype(np.float32)
        img_arr = np.expand_dims(img_arr, axis=0)
        return (aspect_ratio,img_arr)


    # 画像のクロッピング(切り落とし)処理を行います
    # TODO: 処理の妥当性検証
    #
    # input
    #   img  処理前の画像 
    #   aspect_ratio  高さ/幅の割合
    # output
    #   処理結果の画像
    @classmethod
    def crop(self, img, aspect_ratio):
        if aspect_ratio <1:
            w = img.shape[0]
            h = int(w * aspect_ratio)
            if w-h>0:
                offset_h,offset_w=(w-h)//2,0
                img =  K.eval(tf.image.crop_to_bounding_box(img, offset_h,offset_w,h,w))
            else:
                offset_h,offset_w=(h-w)//2,0
                size_h= h - offset_h
                size_w=w
                img =  K.eval(tf.image.crop_to_bounding_box(img, offset_h,offset_w,size_h,size_w))
        else:
            h = img.shape[1]
            w = int(h // aspect_ratio)
            img = K.eval(tf.image.crop_to_bounding_box(img, 0,(h-w)//2,h,w))
        return img


if __name__ == "__main__":
    #filepath="../images/train/img01.jpg"
    filepath="../images/train/img01_2.jpg"
    height,width,channles=ImageUtil.get_image_size(filepath)
    print("width: " + str(width))
    print("height: " + str(height))
    new_height,new_width = 256,256
    file_out="../images/train/img01_2.jpg"
    #ImageUtil.resize_file(filepath, new_height, new_width, file_out)


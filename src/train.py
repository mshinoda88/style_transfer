from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import time,os,argparse
import numpy as np
from imageio import imwrite

from loss import dummy_loss
import nets
from style_util import StyleUtil

# 画像のスタイル変換の学習機能を提供するクラス
class StyleTransferTrainer:
    # コンストラクタ
    #
    # input
    #   dir_img  画像配置ディレクトリ
    #   args 引数情報
    def __init__(self, dir_img, args):
        self.args=args
        self.dir_img   = dir_img
        self.dir_train = self.dir_img + "/train"
        os.makedirs(self.dir_train, exist_ok=True)
        self.dir_output=self.dir_img + "/output"
        os.makedirs(self.dir_output, exist_ok=True)
        self.style_util = StyleUtil("../conf", self.dir_img+"/style")
        self.epochs = args.epochs

    def execute(self, args):
        style_weight= args.style_weight
        content_weight= args.content_weight
        tv_weight= args.tv_weight
        style= args.style
        img_width = img_height = args.image_size

        style_image_path = self.style_util.get_style_img_path(style)

        net = nets.image_transform_net(img_width,img_height,tv_weight)
        model = nets.loss_net(net.output, net.input,
                  img_width, img_height, style_image_path, content_weight, style_weight)
        model.summary()

        #nb_epoch = 82785 *2
        nb_epoch = self.epochs
        train_batchsize =  1

        learning_rate = 1e-3 
        optimizer = Adam() # Adam(lr=learning_rate,beta_1=0.99)

        # Dummy loss since we are learning from regularizes
        model.compile(optimizer, dummy_loss)

        datagen = ImageDataGenerator()
        # Dummy output, not used since we use regularizers to train
        dummy_y = np.zeros((train_batchsize,img_width,img_height,3)) 
        print("dummy_y", type(dummy_y))
        #model.load_weights(style+'_weights.h5',by_name=False)
        print("dir_train:",self.dir_train, img_width, img_height)

        skip_to = 0
        i=0
        t1 = time.time()
        img_ary = datagen.flow_from_directory(
            self.dir_train, target_size=(img_width, img_height), color_mode='rgb',
            classes=None, class_mode=None, batch_size=train_batchsize, 
            shuffle=False, seed=None, save_to_dir=None, save_prefix='', 
            save_format='jpg', follow_links=False, subset=None, interpolation='nearest')
        if len(img_ary)==0:
            print("training images Not Found.:",self.dir_train)
            return

        for x in img_ary:
            print("epoch:",i)
            if i > nb_epoch:
                break

            if i < skip_to:
                i+=train_batchsize
                if i % 1000 ==0:
                    print("skip to: %d" % i)

                continue

            hist = model.train_on_batch(x, dummy_y)
            if i % 10 == 0:
                print(hist,(time.time() -t1))
                t1 = time.time()

            if i % 10 == 0:
                print("epoc: ", i)
                val_x = net.predict(x)

                self._display_img(i, x[0], style)
                self._display_img(i, val_x[0],style, True)
                model.save_weights(style+'_weights.h5')

            i+=train_batchsize


    # save current generated image
    def _display_img(self, i, x, style, is_val=False):
        img = x
        if is_val:
            fname = self.dir_output + '/%s_%d_val.png' % (style,i)
        else:
            fname = self.dir_output + '/%s_%d.png' % (style,i)
        imwrite(fname, img)
        print('Image saved as', fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')
        
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')
          
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='epoch times')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=4.0, type=float)
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    trainer =StyleTransferTrainer("../images", args)
    trainer.execute(args)

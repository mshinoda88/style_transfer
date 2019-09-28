import os,sys,glob

# スタイル関連のユーティリティクラス
class StyleUtil:
    # コンストラクタ
    #
    # input
    #   dir_style_weight  スタイルの事前学習済重み格納ディレクトリ
    #   dir_style_img     スタイル画像格納ディレクトリ
    def __init__(self, dir_style_weight, dir_style_img):
        self.dir_style_weight = dir_style_weight
        self.dir_style_img    = dir_style_img

    # 利用可能なスタイル名の一覧を返します
    #
    # input
    #   なし
    # output
    #   利用可能なスタイル名の配列
    def get_style_list(self):
        ret = []

        filepaths = glob.glob(self.dir_style_weight+'/*_weights.h5')
        for filepath in filepaths:
            basename = os.path.basename(filepath)
            basename = basename.replace("_weights.h5","")
            ret.append(basename)

        return ret

    # 利用可能なスタイルかどうかを判定します
    #
    # input
    #   style チェックするスタイル名
    # output
    #   利用可能ならTrue, 利用不可ならFalse を返します
    def is_valid_style(self, style):
        style_list = self.get_style_list()
        return (style in style_list)

    # 指定されたスタイルの重みを表すファイルパスを返します
    #
    # input
    #   style スタイル名
    # output
    #   スタイルの重みを表すファイルパスを返します
    def get_style_weight_path(self, style):
        path = "{0}/{1}_weights.h5".format(self.dir_style_weight, style)
        return path

    # スタイル画像のパスを返します
    #
    # input
    #   style スタイル名
    # output
    #   スタイル画像のパスを返します
    def get_style_img_path(self, style):
        return self.dir_style_img + "/" + style + ".jpg"


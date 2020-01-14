INPUT_SIZE = 512
BATCH_SIZE = 32
BACKGROUND_RATIO = 3. / 8
# D:\\github_projects\\pixel_text\\training\\
TRAINING_DATA_PATH = 'D:/github_projects/data/icdar_2015/train'

# WINDOWS or LINUX 因为linux和windows平台的路径/和\\是有区别的
PLAT_FORM = 'WINDOWS'
# 是否区分图片后缀名大小写，默认小写（在windows平台上好像不区分，会导致拿到两次）
IS_CASE_IMAGE_SUFFIXES = True
# 裁剪图片最小比例
MIN_CROP_SIDE_RATIO = 0.1
# 图片缩放比例
RANDOM_SCALE = [1]
# 识别四边形还是矩形框 'RBOX' or 'QUAD'
GEOMETRY = 'RBOX'
# 最小文本大小
MIN_TEXT_SIZE = 10
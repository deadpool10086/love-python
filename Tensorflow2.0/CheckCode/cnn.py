import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_data_dir = r'E:\Learn\love-python\Tensorflow2.0\CheckCode\train'
test_data_dir = r'E:\Learn\love-python\Tensorflow2.0\CheckCode\test'
model_dir = r'E:\Learn\love-python\Tensorflow2.0\CheckCode\model.h5'

def denoising(image):
    """
    处理图片，方便更好识别，学习
    :param image:图片对象
    :return: 处理之后的图片
    """

    threshold = 128  # 通过设置阈值，去除不必要的干扰物

    for i in range(image.width):
        for j in range(image.height):
            r,g,b = image.getpixel((i,j))
            if (r > threshold or g >threshold or b > threshold):
                r=255
                g=255
                b=255
                image.putpixel((i,j),(r,g,b))
            else:
                r = 0
                g = 0
                b = 0
                image.putpixel((i, j), (r, g, b))

    # 灰度图片
    image = image.convert('L')
    return image

def gen_train_data(filePath):
    '''
       生成数据集
       :param filePath: 存filePath文件夹获取全部图片处理
       :return: x_data:图片数据，shape=(num, 20, 80),y_data:标签信息, shape=(num, 4)
       '''

    #返回指定的文件夹包含的文件或文件夹的名字的列表。
    train_file_name_list = os.listdir(filePath)
    # 返回值
    x_data = []
    y_data = []

    # 对每个图片单独处理
    for selected_train_file_name in train_file_name_list:
        if selected_train_file_name.endswith('.png'):

            # 获取图片对象
            captcha_image = Image.open(os.path.join(filePath, selected_train_file_name))

            # 对图片去噪，后面对这个方法单独说明
            captcha_image = denoising(captcha_image)
            # captcha_image = captcha_image.convert('L') # 对于简单的不用去噪，灰度反而更有利
            captcha_image_np = np.array(captcha_image)

            # 下面这两个是tensorflow获取图片信息，这里我们还是以上面为例
            # img = tf.io.read_file(os.path.join(filePath, selected_train_file_name))
            # img_np = tf.image.decode_jpeg(img, channels=0)

            img_np = np.array(captcha_image_np)
            # 把每个处理后的数据，塞进x_data,y_data
            x_data.append(img_np)
            y_data.append(np.array(list(selected_train_file_name.split('.')[0])))

    x_data = np.array(x_data).astype(np.float)
    y_data = np.array(y_data)
    return x_data,y_data

# 生成训练集
(x,y) = gen_train_data(train_data_dir)
# 生成测试集
(x_test,y_test) = gen_train_data(test_data_dir)
print(x.shape,y.shape) #(num个图片验证码, 20宽, 80高) (955个图片验证码, 4)


def preprocess(x,y):
    """
    对x,y进行数据处理，转成tensor张量，小范围缩小在-1~1之间
    """
    x = 2*tf.cast(x,dtype=tf.float32)/255.-1
    x = tf.expand_dims(x,-1)
    y = tf.cast(y,dtype=tf.int32)
    return x,y


batch_size = 10
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(1)


model = Sequential([
    # 第一个卷积层
    layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # layers.Dropout(0.25),
    # 第二个卷积层
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # layers.Dropout(0.25),
    layers.Flatten(),

    # 全连接
    layers.Dense(1024),
    layers.Dense(248), # 因为这里我们4个数字，所以也就4*10可能性
    layers.Reshape([4,62])
])

model.build(input_shape=[None, 20, 60, 1])
model.summary()
# 设置学习率
optimizer = optimizers.Adam(lr=1e-3)


def train():
    global model
    # 如果存在模型，就拿以前的继续训练，不用再从头开始
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model('model.h5', compile=False)

    # 进行20次重复训练
    for epoch in range(20):
        for step, (x, y) in enumerate(train_db):
            # 有的时候验证码不是这种格式，就没处理所以就不是的直接过滤
            if x.shape == (10, 20, 80, 1):
                with tf.GradientTape() as tape:
                    # logits
                    logits = model(x)
                    # 真实值就行one_hot编码来对比
                    y_onehot = tf.one_hot(y, depth=10)
                    # 设置loss
                    loss_ce = tf.losses.MSE(y_onehot, logits)
                    loss_ce = tf.reduce_mean(loss_ce)
                # 不断更新梯度
                grads = tape.gradient(loss_ce, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 10 == 0:
                    print(epoch, step, 'loss:', float(loss_ce))
    # 因为一次就已经很高了，所以直接保存模型
    model.save('model.h5')

def test():
    model = tf.keras.models.load_model('model.h5', compile=False)
    for step, (x, y) in enumerate(test_db):
        if x.shape == (1, 20, 80, 1):
            logits = model(x)
            logits = tf.nn.softmax(logits)
            pred = tf.cast(tf.argmax(logits,axis=2),dtype=tf.int32)
            print('预测值：',pred[0].numpy(),'真实值：',y[0].numpy(),'是否相同：',int(tf.reduce_sum(tf.cast(tf.equal(pred,y),dtype=tf.int32)))==4)

if __name__ == '__main__':

    #判断是否存在模型文件，没有则训练生成
    choice_flag = 1 # 0训练 1测试
    if os.path.exists(model_dir) and choice_flag==1:
        test()
    else:
        train()


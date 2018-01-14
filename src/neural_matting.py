import tensorflow as tf
import numpy as np
import os


# Training Parameters
TRY = "1"
BATCH_SIZE = 128
ITER_NUM = int(1e6)


# constant
PROJ_NAME = "matting_nn"
DATA_PATH = "npz_s"


# train data
train_data = []
for npz_file in os.listdir(DATA_PATH):
    train_data.append(np.load(os.path.join(DATA_PATH, npz_file)))
train_data = np.concatenate(train_data, axis=0)
print (train_data.shape)


# function
def pre_process_data(data):
    data = tf.random_crop(data, [27, 27, 6])
    data = tf.image.random_flip_left_right(data)
    # data = tf.image.per_image_standardization(data)
    return data

def random_batch(data):
    idx = np.random.choice(len(data), size=BATCH_SIZE, replace=False)
    return data[idx, :, :, :]


# placeholder 
data = tf.placeholder(tf.float32, [None,32,32,6], name='data')
data_cropped = tf.map_fn(lambda x: pre_process_data(x), data)
X, Y = tf.split(data_cropped, num_or_size_splits=[5, 1], axis=3, name='split_x_y')
img, alphas = tf.split(X, num_or_size_splits=[3, 2], axis=3, name='split_img_alpha')
Y_resized = tf.image.resize_images(Y, [15, 15])
# print(X.get_shape())
# print(Y_resized.get_shape())


# CNN
W_conv1 = tf.Variable(tf.truncated_normal([9,9,5,64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_conv1, strides=[1,1,1,1], padding='VALID'), b_conv1))

W_conv2 = tf.Variable(tf.truncated_normal([1,1,64,64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.0, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_conv2, strides=[1,1,1,1], padding='VALID'), b_conv2))

W_conv3 = tf.Variable(tf.truncated_normal([1,1,64,64], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.0, shape=[64]))
conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_conv3, strides=[1,1,1,1], padding='VALID'), b_conv3))

W_conv4 = tf.Variable(tf.truncated_normal([1,1,64,64], stddev=5e-2))
b_conv4 = tf.Variable(tf.constant(0.0, shape=[64]))
conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, W_conv4, strides=[1,1,1,1], padding='VALID'), b_conv4))

W_conv5 = tf.Variable(tf.truncated_normal([1,1,64,32], stddev=5e-2))
b_conv5 = tf.Variable(tf.constant(0.0, shape=[32]))
conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, W_conv5, strides=[1,1,1,1], padding='VALID'), b_conv5))

W_conv6 = tf.Variable(tf.truncated_normal([5,5,32,1], stddev=5e-2))
b_conv6 = tf.Variable(tf.constant(0.0, shape=[1]))
conv6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, W_conv6, strides=[1,1,1,1], padding='VALID'), b_conv6))


# loss
predict_alpha = conv6
tf.summary.image('input_img',     img[0], max_outputs=1)
tf.summary.image('predict_alpha', predict_alpha[0], max_outputs=1)
tf.summary.image('ground_true',   Y_resized[0], max_outputs=1)
l2_loss = tf.reduce_mean(tf.pow(predict_alpha - Y_resized, 2))
tf.summary.scalar('loss', l2_loss)


# optimizer
global_step = tf.Variable(initial_value=0, trainable=False)
optimize = tf.train.AdamOptimizer().minimize(l2_loss, global_step=global_step)


# saver
# saver = tf.train.Saver()
# save_dir = "checkpoints/"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, PROJ_NAME)


# session
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(os.path.join("logs", PROJ_NAME+TRY), sess.graph)
# try:
#     print("Trying to restore last checkpoint ...")
#     last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
#     saver.restore(sess, save_path=last_chk_path)
#     print("Restored checkpoint from:", last_chk_path)
# except:
#     print("Failed to restore checkpoint. Initializing variables instead.")
sess.run(tf.global_variables_initializer())


# run iteration
for i in range(ITER_NUM):
    batch_data = random_batch(train_data)
    feed_dict = {data: batch_data}
    global_step_, merged_, l2_loss_, optimize_ = sess.run(
        [global_step, merged, l2_loss, optimize], feed_dict=feed_dict)
    writer.add_summary(merged_, global_step_)
    print('loss at step %s: %s' % (global_step_, l2_loss_))

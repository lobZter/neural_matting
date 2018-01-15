import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.preprocessing import normalize


# Training Parameters
TRY = "01"
BATCH_SIZE = 256
ITER_NUM = int(1e6)
DISPLAY_STEP = 100


# constant
PROJ_NAME = "matting_nn"
PROJ_ROOT = "/home/xdex/lobst3rd_workspace/neural_matting"
TRAIN_DATA_PATH = os.path.join(PROJ_ROOT, "dataset", "npy_norm")
TEST_DATA_PATH = os.path.join(PROJ_ROOT, "dataset", "test", "rgb_img")
TEST_GT_PATH = os.path.join(PROJ_ROOT, "dataset", "test", "alpha_gt")
TEST_CF_PATH = os.path.join(PROJ_ROOT, "dataset", "test", "alpha_cf")
TEST_KNN_PATH = os.path.join(PROJ_ROOT, "dataset", "test", "alpha_knn")


# test data
print ("load test data...")
# test_data_np = []
# for img_file in os.listdir(TEST_DATA_PATH):
img_file = "GT01.png"
print (img_file)
# alpha ground-truth
alpha_gt = cv2.imread(os.path.join(TEST_GT_PATH, img_file), cv2.IMREAD_GRAYSCALE)
alpha_gt = alpha_gt.reshape(alpha_gt.shape[0], alpha_gt.shape[1], 1)
alpha_gt = alpha_gt / 255.0
# alpha closed form
alpha_cf = cv2.imread(os.path.join(TEST_CF_PATH, img_file), cv2.IMREAD_GRAYSCALE)
alpha_cf = alpha_cf.reshape(alpha_cf.shape[0], alpha_cf.shape[1], 1)
alpha_cf = alpha_cf / 255.0
# alpha knn
alpha_knn = cv2.imread(os.path.join(TEST_KNN_PATH, img_file), cv2.IMREAD_GRAYSCALE)
alpha_knn = alpha_knn.reshape(alpha_knn.shape[0], alpha_knn.shape[1], 1)
alpha_knn = alpha_knn / 255.0
# test image
rgb_img = cv2.imread(os.path.join(TEST_DATA_PATH, img_file)) / 255.0
# normalize image
rgb_img = rgb_img.reshape((-1, 3))
normalize(rgb_img, norm="l2", axis=1)
rgb_img = rgb_img.reshape((alpha_gt.shape[0],alpha_gt.shape[1], 3))

concat_img = np.concatenate((rgb_img, alpha_knn, alpha_cf, alpha_gt), axis=-1)
test_data_np = np.expand_dims(concat_img, axis=0)
print (test_data_np.shape)
    # test_data_np.append(concat_img)
print ("finish loading test data...")


# train data
print ("load train data...")
train_data_np = []
for npz_file in os.listdir(TRAIN_DATA_PATH):
    train_data_np.append(np.load(os.path.join(TRAIN_DATA_PATH, npz_file)))
train_data_np = np.concatenate(train_data_np, axis=0)
print ("finish loading train data...")
print (train_data_np.shape)


# function
def pre_process_data(data):
    data = tf.random_crop(data, [27, 27, 6])
    data = tf.image.random_flip_left_right(data)
    return data

def pre_process_image(rgb):
    return tf.image.per_image_standardization(rgb)

def random_batch(data):
    idx = np.random.choice(len(data), size=BATCH_SIZE, replace=False)
    return data[idx, :, :, :]

def pad_six(img):
    return tf.pad(img, paddings=[[6, 6], [6, 6]], mode="CONSTANT")


# placeholder
is_train = tf.placeholder(tf.bool)
train_data = tf.placeholder(tf.float32, [None, 32, 32, 6])
test_data = tf.placeholder(tf.float32, [None, None, None, 6])
# pre-process
train_data_cropped = tf.map_fn(lambda x: pre_process_data(x), train_data)
train_X, train_Y = tf.split(train_data_cropped, num_or_size_splits=[5, 1], axis=3)
train_img, train_alphas = tf.split(train_X, num_or_size_splits=[3, 2], axis=3)
# train_img_normalized = tf.map_fn(lambda x: pre_process_image(x), train_img)
# train_X_preprocessed = tf.concat([img_normalized, alphas], axis=3)
test_r, test_g, test_b, test_cf, test_knn, test_Y = tf.split(test_data, num_or_size_splits=[1, 1, 1, 1, 1, 1], axis=3)
test_r = tf.expand_dims(tf.expand_dims(pad_six(tf.squeeze(test_r)), 0), -1)
test_g = tf.expand_dims(tf.expand_dims(pad_six(tf.squeeze(test_g)), 0), -1)
test_b = tf.expand_dims(tf.expand_dims(pad_six(tf.squeeze(test_b)), 0), -1)
test_cf = tf.expand_dims(tf.expand_dims(pad_six(tf.squeeze(test_cf)), 0), -1)
test_knn = tf.expand_dims(tf.expand_dims(pad_six(tf.squeeze(test_knn)), 0), -1)
test_img = tf.concat([test_r, test_g, test_b], axis=3)
test_X = tf.concat([test_r, test_g, test_b, test_cf, test_knn], axis=3)

# CNN
W_conv1 = tf.Variable(tf.truncated_normal([9, 9, 5, 64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.cond(is_train, lambda: tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(train_X, W_conv1, strides=[1,1,1,1], padding='VALID'), b_conv1)), lambda: tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(test_X, W_conv1, strides=[1, 1, 1, 1], padding='VALID'), b_conv1)))

W_conv2 = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.0, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='VALID'), b_conv2))

W_conv3 = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.0, shape=[64]))
conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID'), b_conv3))

W_conv4 = tf.Variable(tf.truncated_normal([1, 1, 64, 64], stddev=5e-2))
b_conv4 = tf.Variable(tf.constant(0.0, shape=[64]))
conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID'), b_conv4))

W_conv5 = tf.Variable(tf.truncated_normal([1, 1, 64, 32], stddev=5e-2))
b_conv5 = tf.Variable(tf.constant(0.0, shape=[32]))
conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, W_conv5, strides=[1, 1, 1, 1], padding='VALID'), b_conv5))

W_conv6 = tf.Variable(tf.truncated_normal([5, 5, 32, 1], stddev=5e-2))
b_conv6 = tf.Variable(tf.constant(0.0, shape=[1]))
conv6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, W_conv6, strides=[1, 1, 1, 1], padding='VALID'), b_conv6))


# loss
predict_alpha = conv6
Y_resized = tf.image.resize_images(train_Y, [15, 15])

def train_loss():
    return tf.reduce_mean(tf.pow(predict_alpha - Y_resized, 2))
def test_loss():
    return tf.reduce_mean(tf.pow(predict_alpha - test_Y, 2))
l2_loss = tf.cond(is_train, lambda: train_loss(), lambda: test_loss())


# summary
# def train_summary():
tf.summary.image('img',   train_img[:1,:,:,:],     max_outputs=1, collections=["train"])
tf.summary.image('alpha', predict_alpha[:1,:,:,:], max_outputs=1, collections=["train"])
tf.summary.image('gt',    Y_resized[:1,:,:,:],     max_outputs=1, collections=["train"])
tf.summary.scalar('loss', l2_loss, collections=["train"])
#     return True
# def test_summary():
tf.summary.image('test_img',   test_img,      max_outputs=27, collections=["test"])
tf.summary.image('test_alpha', predict_alpha, max_outputs=27, collections=["test"])
tf.summary.image('test_gt',    test_Y,        max_outputs=27, collections=["test"])
tf.summary.scalar('test_loss', l2_loss, collections=["test"])
    # return True
# tf.cond(is_train, train_summary, test_summary)
# if tf.equal(is_train, True):
#     train_summary()
# else:
#     test_summary()


# optimizer
global_step = tf.Variable(initial_value=0, trainable=False)
optimize = tf.train.AdamOptimizer().minimize(l2_loss, global_step=global_step)


# saver
saver = tf.train.Saver()
save_dir = "checkpoints/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, PROJ_NAME+TRY)


# session
sess = tf.Session()
train_merged = tf.summary.merge_all(key='train')
test_merged = tf.summary.merge_all(key='test')
train_writer = tf.summary.FileWriter(os.path.join("logs", PROJ_NAME+"_train_"+TRY), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join("logs", PROJ_NAME+"_test_"+TRY), sess.graph)

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
sess.run(tf.global_variables_initializer())


# run iteration
for i in range(ITER_NUM):
    batch_data = random_batch(train_data_np)
    feed_dict = {train_data: batch_data, test_data: test_data_np, is_train: True}
    global_step_, merged_, l2_loss_, optimize_ = sess.run(
        [global_step, train_merged, l2_loss, optimize], feed_dict=feed_dict)
    train_writer.add_summary(merged_, global_step_)
    print('loss at step %s: %s' % (global_step_, l2_loss_))

    if i % DISPLAY_STEP == 0:
        feed_dict = {train_data: batch_data, test_data: test_data_np, is_train: False}
        global_step_, merged_, l2_loss_ = sess.run(
            [global_step, test_merged, l2_loss], feed_dict=feed_dict)
        train_writer.add_summary(merged_, global_step_)
        print('test loss at step %s: %s' % (global_step_, l2_loss_))


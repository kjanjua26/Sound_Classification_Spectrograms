import tensorflow as tf 
import numpy as np 
import glob
from sklearn.model_selection import train_test_split 
import cv2 
import tflearn
from tensorflow.contrib.layers import flatten
import tensorflow.contrib.layers as initializers
import tensorflow.contrib.slim as slim
from sklearn import preprocessing
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
data_path = "urbandataset/dataset/"

X = tf.placeholder(tf.float32, shape=(None,32,32,3), name="input")
Y = tf.placeholder(tf.float32, shape=(None, 9), name="labels")
global_step = tf.Variable(0, trainable=False, name='global_step')
batch_size = 128
num_epochs = 8001
noClasses = 9
def get_data():
	mapping_file = open("mapping_file_res_urban.txt", "w+")
	if os.path.isfile('images_urban.npy') and os.path.isfile('labels_urban.npy'):
		images = np.load('images_urban.npy')
		labels = np.load('labels_urban.npy')
	else:
		images = []
		labels = []
		for i in glob.glob(data_path+"*/*.png"):
			print("For: ", i)
			label = i.split("/")[-2]
			img = cv2.imread(i)
			img = cv2.resize(img, (32,32))
			images.append(normalize(img))
			labels.append(label)
			np.save('images_urban.npy', images)
			np.save('labels_urban.npy', labels)

	assert len(labels) == len(images)
	unique_labels = list(set(labels))
	labelEncoder = preprocessing.LabelEncoder()
	labelEncoder.fit(unique_labels)
	mapping_file.write(str(unique_labels))
	mapping_file.write("\n")
	mapping_file.write(str(list(labelEncoder.transform(unique_labels))))
	mapping_file.close()
	labelEncoder.fit(labels)
	encoded_labels = labelEncoder.transform(labels)
	one_hot_labels = one_hot_encode(encoded_labels)
	x_train, x_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.33, random_state=42)
	x_train = np.asarray(x_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	y_train = np.asarray(y_train)
	print("Done Data Formation.")
	print("Labels Shape: ", y_train.shape)
	return x_train, x_test, y_train, y_test

def next_batch(num, data, labels):
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def one_hot_encode(x):
	encoded = np.zeros((len(x), noClasses))
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	return encoded

def normalize(x):
	min_val = np.min(x)
	max_val = np.max(x)
	x = (x-min_val) / (max_val-min_val)
	return x

def conv_net(x):
	keep_prob = 0.7
	conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
	conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
	conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
	conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))
	conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
	conv1 = tf.nn.relu(conv1)
	conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	conv1_bn = tf.layers.batch_normalization(conv1_pool)
	conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
	conv2 = tf.nn.relu(conv2)
	conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
	conv2_bn = tf.layers.batch_normalization(conv2_pool)
	conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
	conv3 = tf.nn.relu(conv3)
	conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
	conv3_bn = tf.layers.batch_normalization(conv3_pool)
	conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
	conv4 = tf.nn.relu(conv4)
	conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	conv4_bn = tf.layers.batch_normalization(conv4_pool)
	flat = tf.contrib.layers.flatten(conv4_bn)  
	full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
	full1 = tf.nn.dropout(full1, keep_prob)
	full1 = tf.layers.batch_normalization(full1)
	full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
	full2 = tf.nn.dropout(full2, keep_prob)
	full2 = tf.layers.batch_normalization(full2)
	full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
	full3 = tf.nn.dropout(full3, keep_prob)
	full3 = tf.layers.batch_normalization(full3)    
	full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
	full4 = tf.nn.dropout(full4, keep_prob)
	full4 = tf.layers.batch_normalization(full4)        
	out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=noClasses, activation_fn=None)
	return out

def build_network(input_images, labels):
	logits = conv_net(input_images)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
	return loss, accuracy

def train(x_train, x_test, y_train, y_test):
	loss, accuracy = build_network(X, Y)
	optimizer = tf.train.AdamOptimizer(0.0001) 
	train_op = optimizer.minimize(loss, global_step=global_step)
	summary_op = tf.summary.merge_all()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for i in range(num_epochs):
		batch_images, batch_labels = next_batch(batch_size, x_train, y_train)
		_, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={X:batch_images, Y:batch_labels})
		print("Step: {} Train Loss: {} Train Acc: {}".format(i, train_loss, train_acc))
		if i % 1000 == 0:
			val_acc, val_loss = sess.run([accuracy, loss], feed_dict={X:x_test, Y:y_test})
			print("")
			print("Step: {} Val Loss: {} Val Acc: {}".format(i, val_loss, val_acc))
			save_path = saver.save(sess, "model/model-epoch{}.ckpt".format(i))
			print("Model saved for epoch # {}".format(i))
			print("")

def inference(img):
	print("For: ", img)
	img = cv2.imread(img)
	img = cv2.resize(img, (32,32))
	img = normalize(img)
	output = conv_net(X)
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    ckpt = tf.train.get_checkpoint_state("model")
	    saver.restore(sess, "model/model-epoch6000.ckpt")
	    result = sess.run(output, feed_dict={X:[img]})
	    result = tf.nn.softmax(result)
	    inf_result = tf.argmax(result, dimension=1)
	    class_result = sess.run(inf_result)
	    print("Class Result: ", class_result)
		#return class_result


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = get_data()
	train(x_train, x_test, y_train, y_test)
	#inference("31.png")

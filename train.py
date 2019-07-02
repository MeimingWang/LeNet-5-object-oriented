import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from lenet import Lenet

mnist = read_data_sets("mnist_data/", one_hot=True)

batch_size = 100
lenet_part = Lenet(mu = 0, sigma = 0.3, learning_rate = 0.001)
merged = lenet_part.merged_summary
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./log/train', sess.graph)
    test_writer = tf.summary.FileWriter('./log/test')

    for i in range(3000):
        batch = mnist.train.next_batch(batch_size)
        _, train_acc, train_sum= sess.run([lenet_part.training_step, lenet_part.accuracy, merged],
                                    feed_dict = {lenet_part.raw_input_image: batch[0],lenet_part.raw_input_label: batch[1]})

        test_acc, test_sum = sess.run([lenet_part.accuracy, merged],
                                      feed_dict = {lenet_part.raw_input_image: mnist.test.images,
                                                   lenet_part.raw_input_label: mnist.test.labels})
        if i%10 ==0:
            test_writer.add_summary(test_sum, i)
            train_writer.add_summary(train_sum, i)
            print('[train_acc, test_acc]: ', train_acc, test_acc)
    train_writer.close()
    test_writer.close()

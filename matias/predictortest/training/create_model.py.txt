import numpy as np
from sklearn import svm, metrics
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.externals import joblib
import os
import json

if __name__ == '__main__':
    PERSISTENT_VOLUME = '/data'

    # set dump dir based on config_dict env variable
    try:
        json_conf = os.environ['config_dict'].replace('||', '"')
        config = json.loads(json_conf)
        config['pv'] = PERSISTENT_VOLUME
        dump_dir = '{pv}/{model_id}-{model_version}-{owner_id}-{data_version}'.format(**config)
    except KeyError as e:
        print('Missing env variable config_dict or mandatory key in it. %s' % e)
        raise

    data_dir = '%s/tmp/tensorflow/mnist/input_data' % PERSISTENT_VOLUME
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    train_data=mnist.train.images
    train_labels=np.array(np.where(mnist.train.labels==1))[1]

    classifier = svm.SVC(gamma=0.001)
    classifier.fit(train_data, train_labels)

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    fullfilename = '%s/%s' % (dump_dir, 'SVM.pkl')
    try:
        joblib.dump(classifier, fullfilename)
    except Exception as e:
        print('Dump model error: %s' % str(e))
        pass


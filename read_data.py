import tensorflow as tf

filenames=['D:\\wangchen\\tfrecord\\snippetFeature.tfrecord', 'D:\\wangchen\\tfrecord\\snippetFeature.tfrecord']
options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

def parse_function(example_proto):
	dics = {
		'tensor': tf.FixedLenFeature(shape=(), dtype=tf.string),
		'tensor_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
		'label': tf.FixedLenFeature(shape=(), dtype=tf.string)
	}
	parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
	parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.float32)
	parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], parsed_example['tensor_shape'])
	return parsed_example


dataset = tf.data.TFRecordDataset(filenames, compression_type='ZLIB')
new_dataset = dataset.map(parse_function)
iterator = new_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.InteractiveSession()

i=1
while True:
	try:
		tensor, label = sess.run([next_element['tensor'], next_element['label']])
	except tf.errors.OutOfRangeError:
		print('end!')
		break
	else:
		print('============ example %s ============' %i)
		print('tensor: shape: %s | type: %s' %(tensor.shape, tensor.dtype))
		print('label: %s' %(label))
	i = i+1
import os,sys
import tensorflow as tf
import numpy as np
import rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops import array_ops
#//////
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1 # 0 is reserved to space
#//////
num_features=3
num_filters=32
num_hidden=1024
num_layers=3
batch_size=10
num_classes=28
dropout=0.2
global_step=0
file_number=10
initial_lr=0.0001
step=0
#///////
strokes = np.load('strokes.npy')
with open('sentences.txt') as f:
    texts = f.readlines()

def dynamic_lr(initial_lr):
	return initial_lr*0.999
def build_multi_dynamic_brnn(maxTimeSteps,
                             inputX,
                             seqLengths,
                             time_major=True):
    hid_input = inputX
    for i in range(num_layers):
        scope = 'DBRNN_' + str(i + 1)
        forward_cell = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
        backward_cell =tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, output_states = bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLengths,
                                                           time_major=True,
                                                           scope=scope)
        # forward output, backward ouput
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        # forward states, backward states
        output_state_fw, output_state_bw = output_states
        # output_fb = tf.concat(2, [output_fw, output_bw])
        output_fb = array_ops.concat(outputs,2)#tf.concat([output_fw, output_bw], 2)
        shape = output_fb.get_shape().dims
        output_fb = tf.reshape(output_fb, [-1, shape[1].value, 2, int(shape[2].value / 2)])
        hidden = tf.reduce_sum(output_fb, 2)
	hidden=tf.contrib.layers.dropout(hidden, keep_prob=1-dropout,is_training=True)
        #hidden = dropout(hidden, args.keep_prob, (args.mode == 'train'))

        if i != num_layers - 1:
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, num_hidden])
            # output_list = tf.split(0, maxTimeSteps, outputXrs)
            #output_list = tf.split(outputXrs, maxTimeSteps, 0)
            #fbHrs = [tf.reshape(t, [batch_size, num_hidden]) for t in output_list]
    return outputXrs
def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
def prepare_batch(index):
	no_example=0
	data=[]
	text=[]
	length=[]
	original_text=[]
	while(no_example<10):
		current_number=index+no_example
		data.append(strokes[current_number])
		original = ' '.join(texts[current_number].strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '').replace('&', '').replace(';', '').replace(')', '').replace('(', '').replace('-', '').replace(':','').replace('#','').replace('"','').replace('0',' zero ').replace('1',' one ').replace('2',' two ').replace('3',' three ').replace('4',' four ').replace('5',' five ').replace('6',' six ').replace('7',' seven ').replace('8',' eight ').replace('9',' nine ').replace('/','').replace('+','')
		original_text.append(original)
		targets = original.replace(' ', '  ')
		targets = targets.split(' ')
		# Adding blank label
		targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
		# Transform char into index
		targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
		text.append(targets)
		length.append(strokes[current_number].shape[0])
		no_example+=1
	max_len=np.max(length)
	
	for i in range(len(data)):
		if(data[i].shape[0] != max_len):
			
			data[i]=np.pad(data[i],((0,max_len-length[i]),(0,0)),'constant', constant_values=(0, 0))
			#length[i]=data[i].shape[0]
			
	data=np.asarray(data)
	train_inputs = (data - np.mean(data))/np.std(data)
	train_inputs=np.expand_dims(train_inputs, axis=3)
	train_targets =sparse_tuple_from(text)
	train_seq_len=length
	return (train_inputs,train_seq_len,train_targets,original_text)
def model(epoch):
	inputs = tf.placeholder(tf.float32, [None, None, num_features,1])
	seq_len = tf.placeholder(tf.int32, [None])
	learning_rate = tf.placeholder(tf.float32, shape=[])
	targets=tf.sparse_placeholder(tf.int32)
	conved_seq_lens=rnn.get_rnn_seqlen(seq_len)
	kernel= tf.get_variable("conv1",[21, 1, 1,num_filters] ,initializer=None, dtype= tf.float32)
	conv1= tf.nn.conv2d(inputs, kernel,[1, 2, 1, 1],padding='VALID')
	
	#kernel1= tf.get_variable("conv2",[11, 1, num_filters,num_filters] ,initializer=None, dtype= tf.float32)
	#conv2= tf.nn.conv2d(conv1, kernel1,[1, 1, 1, 1],padding='VALID')
	
	fdim = conv1.get_shape().dims
	feat_dim = fdim[2].value * fdim[3].value
	rnn_input = tf.reshape(conv1, [batch_size, -1, feat_dim])
	rnn_input=tf.transpose(rnn_input,(1,0,2))
	print(rnn_input.shape)
	rnn_outputs=build_multi_dynamic_brnn(conved_seq_lens,rnn_input,conved_seq_lens)
	W=tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
	b=tf.Variable(tf.constant(0.0,shape=[num_classes]))
	logits=tf.matmul(rnn_outputs,W)+b
	logits=tf.reshape(logits,[-1,batch_size,num_classes])
	loss=tf.nn.ctc_loss(targets,logits,conved_seq_lens,time_major=True,ignore_longer_outputs_than_inputs=True)
	cost=tf.reduce_mean(loss)
	first_summary = tf.summary.scalar(name='cost_summary', tensor=cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	gvs = optimizer.compute_gradients(cost)
	capped_gvs = [(tf.clip_by_value(grad, -4., 4.), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)
	decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, conved_seq_lens)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
	lr=initial_lr
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		#sess.run(tf.global_variables_initializer())
		saver.restore(sess,"/home/sirena/sumuk/text recognition/model/model.ckpt")
		writer=tf.summary.FileWriter('/home/sirena/sumuk/text recognition/graphs', sess.graph)
		kl_ep=0
		'''for j in range(epoch):
			ct=0
			for i in range(0,5000,10):
				if(kl_ep!=0 and kl_ep%2000==0):
					lr=dynamic_lr(lr)
				#print(i)
				train_inputs,train_seq_len,train_target,orginal_text=prepare_batch(i)
				print(sess.run(loss,feed_dict={inputs:train_inputs,seq_len:train_seq_len,targets:train_target,learning_rate:lr}).shape)
				c,_,f_c=sess.run([cost,train_op,first_summary],feed_dict={inputs:train_inputs,seq_len:train_seq_len,targets:train_target,learning_rate:lr})	
				#print(c)
				writer.add_summary(f_c,kl_ep)
				kl_ep+=1
				ct+=c
			ct=ct/500.0
			print("cost:",ct)
			saver.save(sess, "/home/sirena/sumuk/text recognition/model/model.ckpt")
			train_inputs,train_seq_len,train_target,orginal_text=prepare_batch(0)
			d = sess.run(decoded[0], feed_dict={inputs:train_inputs,seq_len:train_seq_len,targets:train_target,learning_rate:lr})
			ele=0
			for el_nu in xrange(d[0].shape[0]):
				if d[0][el_nu][0] ==0:
					ele+=1
			decode_ele=np.asarray(d[1])[0:ele]
			str_decoded = ''.join([chr(x) for x in decode_ele+ FIRST_INDEX]) #np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

			print('Original val: %s' %orginal_text[0] )
			print('Decoded val: %s' % str_decoded)#'''
		k=np.random.randint(5000,6000-10)
		train_inputs,train_seq_len,train_target,orginal_text=prepare_batch(k)
		d = sess.run(decoded[0], feed_dict={inputs:train_inputs,seq_len:train_seq_len,targets:train_target,learning_rate:lr})
		#print(d)
		start_ele=0
		ele=0															
		for val_i in range(10):
						
				for el_nu in xrange(d[0].shape[0]):
					if d[0][el_nu][0] ==val_i:
						ele+=1
				#print(start_ele,ele)
				decode_ele=np.asarray(d[1])[start_ele:ele]
				start_ele=ele
				str_decoded = ''.join([chr(x) for x in decode_ele+ FIRST_INDEX]) #np.asarray(d[1]) + FIRST_INDEX])
				# Replacing blank label to none
				str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
				print("sample :%s",str(val_i))
				print('Original val: %s' %orginal_text[val_i] )
				print('Decoded val: %s' % str_decoded)#'''
model(10)
'''for i in range(0,5000):
	#train_inputs,train_seq_len,train_target,orginal_text=prepare_batch(i)
	original = ' '.join(texts[i].strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '').replace('&', '').replace(';', '').replace(')', '').replace('(', '').replace('-', '').replace(':','').replace('#','').replace('"','')
	#original_text.append(original)
	targets = original.replace(' ', '  ')
	targets = targets.split(' ')
	# Adding blank label
	targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])
	# Transform char into index
	targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])
	if(np.min(targets)<0 or np.max(targets)>28):
		print(i,original)
	#print(targets)'''

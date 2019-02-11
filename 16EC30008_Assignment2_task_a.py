import numpy as np
import data_loader
import module
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
from sklearn import model_selection
import sys

if sys.argv[1] == '--train':
	exec_mode = 'train'
else:
	exec_mode = 'test'


'''Implement mini-batch SGD here'''
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

batch_size = 64
data_l = data_loader.DataLoader()

if exec_mode == 'train':
	imgs, labels = data_l.load_data(mode = 'train')
	train_imgs, val_imgs, train_labels, val_labels = model_selection.train_test_split(imgs, labels, test_size = 0.3, random_state = 1, stratify = labels)
	train_set = mx.gluon.data.dataset.ArrayDataset(train_imgs, train_labels)
	val_set = mx.gluon.data.dataset.ArrayDataset(val_imgs, val_labels)
	train_data = mx.gluon.data.DataLoader(train_set,batch_size, shuffle=True)
	val_data = mx.gluon.data.DataLoader(val_set,batch_size, shuffle=True)
else:
	imgs, labels = data_l.load_data(mode = 'test')
	test_set = mx.gluon.data.dataset.ArrayDataset(imgs, labels)
	test_data = mx.gluon.data.DataLoader(test_set,
									 batch_size, shuffle=False)

net = module.NN()
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
print('Network 1\n')
if exec_mode == 'train':
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

	epochs = 50
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			# print(data.astype('float32'))
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('./weights/best_model_a1.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)


	line1, = ax.plot(np.arange(len(train_loss)) + 1, train_loss, color = 'r', lw=2)
	line2, = ax.plot(np.arange(len(val_loss)) + 1, val_loss, color = 'g', lw=2)
	ax.legend((line1, line2), ('Training Loss', 'Validation Loss'))
	plt.rcParams.update({'font.size': 15})
	plt.title('Loss vs epochs for Task(a) Network 1')
	plt.rcParams.update({'font.size': 12})
	plt.xlabel('Epoch')
	plt.rcParams.update({'font.size': 12})
	plt.ylabel('Loss')
	plt.xticks([item for item in (np.arange(len(train_loss)) + 1).tolist() if item % 10 == 0])
	plt.savefig('Loss_vs_epochs_a1.png')
	# plt.show()

else:
	net.load_parameters('./weights/best_model_a1.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')

net = module.NN2()
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
print('\nNetwork 2\n')
if exec_mode == 'train':
	# net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

	epochs = 50
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('./weights/best_model_a2.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)


	line1, = ax.plot(np.arange(len(train_loss)) + 1, train_loss, color = 'r', lw=2)
	line2, = ax.plot(np.arange(len(val_loss)) + 1, val_loss, color = 'g', lw=2)
	ax.legend((line1, line2), ('Training Loss', 'Validation Loss'))
	plt.rcParams.update({'font.size': 15})
	plt.title('Loss vs epochs for Task(a) Network 2')
	plt.rcParams.update({'font.size': 12})
	plt.xlabel('Epoch')
	plt.rcParams.update({'font.size': 12})
	plt.ylabel('Loss')
	plt.xticks([item for item in (np.arange(len(train_loss)) + 1).tolist() if item % 10 == 0])
	plt.savefig('Loss_vs_epochs_a2.png')
	# plt.show()

else:
	net.load_parameters('./weights/best_model_a2.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')

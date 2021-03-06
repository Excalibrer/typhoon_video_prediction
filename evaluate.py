import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from make_gif import create_gif

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from settings import *

import warnings

warnings.filterwarnings('ignore')

n_plot = 40
batch_size = 15
nt = 15

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_model.json')
test_file = os.path.join(DATA_DIR, 'Typhoons_X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'Typhoons_sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = 10
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

loss = np.abs(X_test - X_hat)

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

# def store_img(matrix, name):
#     for b, batch in enumerate(matrix):
#         dir_path = os.path.join(RESULTS_SAVE_DIR, name+'plot_'+str(b)+'/')
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         for t, img in enumerate(batch):
#             plt.figure(figsize=(5,5))
#             plt.imshow(img)
#             plt.axis('off')
#             plt.title('time '+str(t))
#             plt.savefig(dir_path+('/plot_%2d.jpg' % t))
#             plt.clf()
#
# store_img(X_test, 'X_test')
# store_img(X_hat, 'X_hat')
#
# def gif_result(test_m, hat_m):
#     for b, (test_batch, hat_batch) in enumerate(zip(test_m, hat_m)):
#         dir_path = os.path.join(RESULTS_SAVE_DIR, 'plot_' + str(b) + '/')
#         if not os.path.exists(dir_path):
#             os.mkdir(dir_path)
#         for t, (test_img, hat_img) in enumerate(zip(test_batch, hat_batch)):
#             plt.figure(figsize=(10, 5))
#             plt.subplot(121)
#             plt.imshow(test_img)
#             plt.axis('off')
#             plt.subplot(122)
#             plt.imshow(hat_img)
#             plt.axis('off')
#             plt.suptitle('time '+str(t))
#             plt.savefig(dir_path+('/plot_%02d.jpg' % t))
#             plt.clf()
#         create_gif(os.path.join(RESULTS_SAVE_DIR, 'plot'+str(b)+'.gif'),
#                    os.path.join(RESULTS_SAVE_DIR, 'plot_' + str(b) + '/'), duration=0.5)
#
# gif_result(X_test, X_hat)
#
# # Plot some predictions
# aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
# plt.figure(figsize = (nt, 3*aspect_ratio))
# gs = gridspec.GridSpec(3, nt)
# gs.update(wspace=0., hspace=0.)
# plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
# if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
# plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
# for i in plot_idx:
#     for t in range(nt):
#         plt.subplot(gs[t])
#         plt.imshow(X_test[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Actual', fontsize=10)
#
#         plt.subplot(gs[t + nt])
#         plt.imshow(X_hat[i,t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
#         if t==0: plt.ylabel('Predicted', fontsize=10)
#
#         plt.subplot(gs[t + 2 * nt])
#         plt.imshow(loss[i, t], interpolation='none')
#         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
#                         labelleft='off')
#         if t == 0: plt.ylabel('Loss', fontsize=10)
#
#     plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
#     plt.clf()

def store_img(matrix, name):
    for b, batch in enumerate(matrix):
        dir_path = os.path.join(RESULTS_SAVE_DIR, name+'pred_plot_'+str(b)+'/')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for t, img in enumerate(batch):
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.axis('off')
            plt.title('time '+str(t))
            plt.savefig(dir_path+('/pred_plot_%2d.jpg' % t))
            plt.clf()

store_img(X_test, 'X_test')
store_img(X_hat, 'X_hat')

def gif_result(test_m, hat_m):
    for b, (test_batch, hat_batch) in enumerate(zip(test_m, hat_m)):
        dir_path = os.path.join(RESULTS_SAVE_DIR, 'pred_plot_' + str(b) + '/')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for t, (test_img, hat_img) in enumerate(zip(test_batch, hat_batch)):
            plt.figure(figsize=(10, 5))
            if t >= layer_config['extrap_start_time']:
                plt.suptitle('time ' + str(t) + ' without actual inputs', fontsize=20)
                plt.subplot(121)
                plt.imshow(test_img)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('Actual')
            else:
                plt.suptitle('time ' + str(t) + ' with actual inputs', fontsize=20)
                plt.subplot(121)
                plt.imshow(test_img)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('Actual')
            plt.subplot(122)
            plt.imshow(hat_img)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Predicted')
            plt.savefig(dir_path+('/pred_plot_%02d.jpg' % t))
            plt.clf()
        create_gif(os.path.join(RESULTS_SAVE_DIR, 'pred_plot'+str(b)+'.gif'),
                   os.path.join(RESULTS_SAVE_DIR, 'pred_plot_' + str(b) + '/'), duration=0.5)

gif_result(X_test, X_hat)

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 3*aspect_ratio))
gs = gridspec.GridSpec(3, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'pred_prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

        plt.subplot(gs[t + 2 * nt])
        plt.imshow(loss[i, t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
                        labelleft='off')
        if t == 0: plt.ylabel('Loss', fontsize=10)

    plt.savefig(plot_save_dir +  'pred_plot_' + str(i) + '.png')
    plt.clf()
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pygments.lexers import get_lexer_by_name
from pygments import lex

import os.path
import argparse
import time
import os
from six.moves import cPickle

from utils.text_loader import TextLoader
from utils.distribution_stats import DistributionStats
from model import Model

from six import text_type

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=str, default='save',
					   help='model directory to store checkpointed models')
	parser.add_argument('--token_file', type=str, default='test.txt',
					   help = 'token file to find entropy values for')
	parser.add_argument('--token_type_file', type=str, default='test_types.txt',
					   help = 'token type file corresponding to the given token file')
	parser.add_argument('--out_file', type=str, default='entropy_stats.pkl',
					   help='File basename to be stored in SAVE_DIR. Must be a .pkl file')

	args = parser.parse_args()
	validate_args(args)
	get_entropy_stats_per_type(args)

def validate_args(args):
	assert os.path.isdir(args.save_dir), "SAVE_DIR {0} doesn't exist".format(args.save_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(args.token_file),"TOKEN_FILE {0} doesn't exist".format(args.token_file)
	assert os.path.isfile(args.token_type_file),"TOKEN_TYPE_FILE {0} doesn't exist".format(args.token_type_file)
	assert args.out_file.endswith('.pkl'), 'OUT_FILE must have extension .pkl'

def get_entropy_stats_per_type(args):
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		(saved_args, reverse_input) = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)

	model = Model(saved_args, reverse_input, True)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			entropy_stats_map = model.get_entropy_stats_per_type(sess, chars, 
				vocab, args.token_file, args.token_type_file)
			with open(os.path.join(args.save_dir, args.out_file), 'wb') as f:
				cPickle.dump(entropy_stats_map, f)

			for token_type, dist_stats in entropy_stats_map.items():
				print("Type: {0}".format(token_type))
				print("\tMean: {0}".format(dist_stats.mean))
				print("\tStd: {0}".format(dist_stats.sd))
				print("\tQ1: {0}".format(dist_stats.q1))
				print("\tQ2: {0}".format(dist_stats.q2))
				print("\tQ3: {0}".format(dist_stats.q3))

if __name__ == '__main__':
	main()

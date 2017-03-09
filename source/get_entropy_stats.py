from __future__ import print_function
import numpy as np
import tensorflow as tf

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
	parser.add_argument('--out_file', type=str, default='entropy_stats.pkl',
					   help='File basename to be stored in SAVE_DIR. Must be a .pkl file')

	args = parser.parse_args()
	validate_args(args)
	get_entropy_stats(args)

def validate_args(args):
	assert os.path.isdir(args.save_dir), "SAVE_DIR {0} doesn't exist".format(args.save_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(args.token_file),"TOKEN_FILE {0} doesn't exist".format(args.token_file)
	assert args.out_file.endswith('.pkl'), 'OUT_FILE must have extension .pkl'

def get_entropy_stats(args):
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
			entropy_dist_stats = model.get_entropy_stats(sess, chars, 
				vocab, args.token_file)
			with open(os.path.join(args.save_dir, args.out_file), 'wb') as f:
				cPickle.dump(entropy_dist_stats, f)

if __name__ == '__main__':
	main()

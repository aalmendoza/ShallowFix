from __future__ import print_function
import numpy as np
from lexer import simplePyLex
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

UNK_TOKEN = '<UNK>'

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', type=str, default='save',
					   help='model directory to store checkpointed models')
	parser.add_argument('--source', type=str, default='code.c',
					   help='source file to evaluate')
	parser.add_argument('--pre_tokenized', type=str, default='false',
					   help='boolean indicating if the source file is already tokenized')
	parser.add_argument('--entropy_stats', type=str,
					   help='entropy distribution statistics .pkl file')

	args = parser.parse_args()
	validate_args(args)
	get_zscores(args)

def validate_args(args):
	assert os.path.isdir(args.save_dir), "data_dir {0} doesn't exist".format(args.data_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(args.entropy_stats), "entropy stats file {0} doesn't exist".format(args.entropy_stats)

def get_zscores(args):
	pre_tokenized = str2bool(args.pre_tokenized)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		(saved_args, reverse_input) = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)
	with open(args.entropy_stats, 'rb') as f:
		entropy_dist_stats = cPickle.load(f)

	if not reverse_input:
		start_token = '<START>'
		end_token = '<EOF>'
	else:
		start_token = '<EOF>'
		end_token = '<START>'

	model = Model(saved_args, reverse_input, True)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(tf.global_variables())
		ckpt = tf.train.get_checkpoint_state(args.save_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			if pre_tokenized:
				token_file = args.source
			else:
				token_file = create_token_file(args.source, args.save_dir, vocab,
					start_token, end_token)
			[entropy_list, zscore_list] = model.get_zscores(sess, chars, vocab,
				token_file, entropy_dist_stats)
			display_results(token_file, entropy_list, zscore_list, entropy_dist_stats)

def str2bool(s):
	return s.lower() in ('t', 'true', '1', 'yes')

def create_token_file(source_file, save_dir, vocab, start_token, end_token):
	token_file = '/tmp/tokenized.txt'
	tmp_outfile = '/tmp/out.txt'
	simplePyLex.main(source_file, tmp_outfile, 3, "full", "True", "False")
	with open(tmp_outfile, 'r') as f, open(token_file, 'w') as token_out:
		token_out.write(start_token + "\n")
		tokens = f.read().split()
		for token in tokens:
			if token in vocab:
				token_out.write(token + "\n")
			else:
				token_out.write(UNK_TOKEN + "\n")
		token_out.write(end_token + "\n")
	return token_file

def display_results(token_file, entropy_list, zscore_list, entropy_dist_stats):
	print("Token                Entropy    Z-Score     Outlier")
	print("====================================================")
	with open(token_file, 'r') as f:
		token = f.readline()[:-1]
		for i in range(len(entropy_list)):
			token = f.readline()[:-1]
			entropy = entropy_list[i]
			zscore = zscore_list[i]
			is_outlier = entropy_dist_stats.is_outlier(entropy)
			print("{0:<20} {1:<.4f}     {2:<3.4f}      {3}".format(token, entropy, zscore, is_outlier))


if __name__ == '__main__':
	main()

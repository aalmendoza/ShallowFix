from __future__ import print_function
import numpy as np
from lexer import simplePyLex
import tensorflow as tf
import csv
from collections import defaultdict

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
	parser.add_argument('--save_dir', type=str, default='../save',
					   help='model directory to store checkpointed models')

	args = parser.parse_args()
	validate_args(args)
	b_f_eval(args)

def validate_args(args):
	assert os.path.isdir(args.save_dir), "save_dir {0} doesn't exist".format(args.save_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.save_dir

def b_f_eval(args):
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
			with open('../data/student/buggy_fix_ce_list_updated.csv', 'r') as csvfile:
				reader = csv.reader(csvfile)
				next(reader, None) # Skip header line
				c = 0
				for row in reader:
					c+=1
					print(c)

					buggy_file_name = row[0]
					entropy_file_name = buggy_file_name + ".entropy.pkl"

					buggy_tokens = get_tokens(buggy_file_name)

					if len(buggy_tokens) > 0:
						buggy_vocab_tokens = convert_to_vocab(buggy_tokens, vocab,
							model.start_token, model.end_token)
						buggy_entropy_scores = model.get_entropy(sess, chars, vocab,
							buggy_vocab_tokens)
						with open(entropy_file_name, 'wb') as f:
							cPickle.dump(buggy_entropy_scores, f)


def get_tokens(source_file):
	with open(source_file, 'r') as f:
		tokens = f.read().split()
	return tokens

def convert_to_vocab(tokens, vocab, start_token, end_token):
	new_vocab = [start_token]
	for token in tokens:
		if token in vocab:
			new_vocab.append(token)
		else:
			new_vocab.append(UNK_TOKEN)
	new_vocab.append(end_token)
	return new_vocab

if __name__ == '__main__':
	main()

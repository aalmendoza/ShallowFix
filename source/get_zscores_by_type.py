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
	get_zscores_by_type(args)

def validate_args(args):
	assert os.path.isdir(args.save_dir), "data_dir {0} doesn't exist".format(args.data_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.init_from
	assert os.path.isfile(args.entropy_stats), "entropy stats file {0} doesn't exist".format(args.entropy_stats)

def get_zscores_by_type(args):
	pre_tokenized = str2bool(args.pre_tokenized)
	with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
		(saved_args, reverse_input) = cPickle.load(f)
	with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
		chars, vocab = cPickle.load(f)
	with open(args.entropy_stats, 'rb') as f:
		entropy_stats_map = cPickle.load(f)

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
				token_type_file = '' # TODO: FIX
			else:
				token_file, token_type_file = create_token_and_type_file(args.source,
					args.save_dir, vocab, start_token, end_token)
			[entropy_list, zscore_list] = model.get_zscores_by_type(sess, chars, vocab,
				token_file, token_type_file, entropy_stats_map)
			display_results(token_file, token_type_file, entropy_list, zscore_list, entropy_stats_map)

def str2bool(s):
	return s.lower() in ('t', 'true', '1', 'yes')

def create_token_and_type_file(source_file, save_dir, vocab, start_token, end_token):
	lexer = get_lexer_by_name('C')
	token_file = '/tmp/tokenized.txt'
	token_type_file = '/tmp/tokenized_types.txt'
	tmp_outfile = '/tmp/out.txt'
	simplePyLex.main(source_file, tmp_outfile, 3, "full", "True", "False")
	with open(tmp_outfile, 'r') as f:
		tokens = f.read().split()	

	with open(token_file, 'w') as token_out, open(token_type_file, 'w') as token_type_out:
		token_out.write(start_token + "\n")
		token_type_out.write('START_TOKEN' + "\n")
		for token in tokens:
			token_type_out.write(get_token_type(lexer, token, start_token, end_token) + "\n")
			if token in vocab:
				token_out.write(token + "\n")
			else:
				token_out.write(UNK_TOKEN + "\n")
		token_out.write(end_token + "\n")
		token_type_out.write("END_TOKEN" + "\n")
	return token_file, token_type_file

def get_token_type(lexer, token, start_token, end_token):
	if token == start_token:
		return 'START_TOKEN'
	elif token == end_token:
		return 'END_TOKEN'
	elif token == UNK_TOKEN:
		return 'Token.Name'
	elif token == '<int>':
		return 'Token.Literal.Number.Integer'
	elif token == '<float>':
		return 'Token.Literal.Number.Float'
	elif token == '<oct>':
		return 'Token.Literal.Number.Oct'
	elif token == '<bin>':
		return 'Token.Literal.Number.Bin'
	elif token == '<hex>':
		return 'Token.Literal.Number.Hex'
	elif token == '<num>':
		return 'Token.Literal.Number'
	elif token == '<str>':
		return 'Token.Literal.String'
	else:
		res = lex(token, lexer)
		return str(list(res)[0][0])

def display_results(token_file, token_type_file, entropy_list, zscore_list, entropy_stats_map):
	print("Token                Entropy    Z-Score     Outlier")
	print("====================================================")
	with open(token_file, 'r') as f_token, open(token_type_file, 'r') as f_type:
		token = f_token.readline()[:-1]
		token_type = f_type.readline()[:-1]
		for i in range(len(entropy_list)):
			token = f_token.readline()[:-1]
			token_type = f_type.readline()[:-1]
			entropy = entropy_list[i]
			zscore = zscore_list[i]
			is_outlier = entropy_stats_map[token_type].is_outlier(entropy)
			print("{0:<20} {1:<.4f}     {2:<3.4f}      {3}".format(token, entropy, zscore, is_outlier))


if __name__ == '__main__':
	main()

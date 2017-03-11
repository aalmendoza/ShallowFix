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
	parser.add_argument('--by_type', type=str, default='false',
					   help='Classify outliers according to sample mean and std per token type')
	parser.add_argument('--out_csv', type=str, default='b_f_stats.csv')
	parser.add_argument('--granularity', type=str, default='line',
					   help='granularity to check for outliers. Either line or token')

	args = parser.parse_args()
	validate_args(args)
	b_f_eval(args)

def validate_args(args):
	assert (args.granularity == 'line' or args.granularity == 'token'), 'Granularity must be token or line'
	assert os.path.isdir(args.save_dir), "save_dir {0} doesn't exist".format(args.save_dir)
	assert os.path.isfile(os.path.join(args.save_dir,"config.pkl")),"config.pkl file does not exist in path %s"%args.save_dir
	assert os.path.isfile(os.path.join(args.save_dir,"chars_vocab.pkl")),"chars_vocab.pkl file does not exist in path %s"%args.save_dir

def b_f_eval(args):
	by_type = str2bool(args.by_type)

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

			# TRUE POSITIVE: Changed and outlier
			# FALSE POSITIVE: Not changed and outlier
			# TRUE NEGATIVE: Not changed and not outlier
			# FALSE NEGATIVE: Changed and not outlier
			ce_type_tp = defaultdict(int)
			ce_type_fp = defaultdict(int)
			ce_type_tn = defaultdict(int)
			ce_type_fn = defaultdict(int)
			ce_type_freq = defaultdict(int)
			total_tp = 0
			total_fp = 0
			total_tn = 0
			total_fn = 0
			keys = set()

			token_level = (args.granularity == 'token')
			if token_level:
				diff_ext = '.diff_token.pkl'
			else:
				diff_ext = '.diff_line.pkl'

			# c = 0
			in_csv = '../data/student/buggy_fix_ce_list_updated.csv'
			with open(in_csv, 'r') as in_csvfile:
				reader = csv.reader(in_csvfile)
				next(reader, None) # Skip header line
				
				for row in reader:
					# c+=1
					# print(c)

					buggy_file_name = row[0]
					diff_file_name = buggy_file_name + diff_ext
					entropy_file_name = buggy_file_name + ".entropy.pkl"
					fix_file_name = row[1]
					ce = row[2]
					ce_type_freq[ce] += 1

					# print("Buggy: {0}".format(buggy_file_name))
					# print("Fixed: {0}".format(fix_file_name))
					# print(ce)
					buggy_tokens = get_tokens(buggy_file_name)
					fixed_tokens = get_tokens(fix_file_name)

					if by_type:
						buggy_token_types = get_token_types(buggy_tokens, model.start_token, model.end_token)
						fixed_token_types = get_token_types(fixed_tokens, model.start_token, model.end_token)

					# Get array corresponding to which tokens were changed
					if (len(buggy_tokens) > 0 and len(fixed_tokens) > 0):
						keys.add(ce)

						with open(diff_file_name, 'rb') as f:
							changed = cPickle.load(f)
						with open(entropy_file_name, 'rb') as f:
							buggy_entropy_scores = cPickle.load(f)

						if token_level:
							buggy_outliers = get_token_outliers(buggy_entropy_scores)
						else:
							buggy_outliers = get_line_outliers(buggy_entropy_scores, buggy_file_name)

						# if by_type:
						# 	buggy_outliers = get_outliers_by_type(buggy_entropy_scores, 
						# 		buggy_token_types, entropy_dist_stats)
						# else:
						# 	buggy_outliers = get_outliers(buggy_entropy_scores, entropy_dist_stats)

						assert len(changed) == len(buggy_outliers), "buggy: {0}".format(buggy_file_name)
						for i in range(len(changed)):
							if changed[i]:
								if buggy_outliers[i]:
									ce_type_tp[ce] += 1
									total_tp += 1
								else:
									ce_type_fn[ce] += 1
									total_fn += 1
							else:
								if buggy_outliers[i]:
									ce_type_fp[ce] += 1
									total_fp += 1
								else:
									ce_type_tn[ce] += 1
									total_tn += 1

						# display_results(buggy_tokens, fixed_tokens, buggy_outliers, changed)
						# print("===========================================")

			with open(args.out_csv,  'w') as out_csvfile:
				writer = csv.writer(out_csvfile, delimiter=',')
				writer.writerow(['Error', 'Frequency', 'Accuracy', 'True Positives', 'False Positives',
						'True Negatives', 'False Negatives', 'Sensitivity (TPR)',
						'Specificity (TNR)', 'Precision'])

				for key in keys:
					tp = ce_type_tp[key]
					fp = ce_type_fp[key]
					tn = ce_type_tn[key]
					fn = ce_type_fn[key]
					p = tp + fn
					n = tn + fp

					if p != 0:
						sensitivity = tp / p # True positive rate
					else:
						sensitivity = 'Undefined'
					if n != 0:
						specificity = tn / n # True negative rate
					else:
						specificity = 'Undefined'
					if tp + fp != 0:
						precision = tp / (tp + fp) # Positive predictive value
					else:
						precision = 'Undefined'

					total = tp + fp + tn + fn
					acc = float(tp + tn) / total
					freq = ce_type_freq[key]

					writer.writerow([key, freq, acc, tp, fp, tn, fn, sensitivity, specificity, precision])

					# print(key)
					# print("\tAccuracy: {0}".format(acc))
					# print("\tTrue Positive: {0}".format(tp))
					# print("\tFalse Positive: {0}".format(fp))
					# print("\tTrue Negative: {0}".format(tn))
					# print("\tFalse Negative: {0}".format(fn))
					# print("\tSensitivity (TPR): {0}".format(sensitivity))
					# print("\tSpecificity (TNR): {0}".format(specificity))
					# print("\tPrecision: {0}".format(precision))
			
			p = total_tp + total_fn
			n = total_tn + total_fp
			if p != 0:
				sensitivity = total_tp / p # True positive rate
			else:
				sensitivity = 'Undefined'
			if n != 0:
				specificity = total_tn / n # True negative rate
			else:
				specificity = 'Undefined'
			if total_tp + total_fp != 0:
				precision = total_tp / (total_tp + total_fp) # Positive predictive value
			else:
				precision = 'Undefined'
			acc = float(total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)

			print("Total Accuracy: {0}".format(acc))
			print("Total True Positive: {0}".format(total_tp))
			print("Total False Positive: {0}".format(total_fp))
			print("Total True Negative: {0}".format(total_tn))
			print("Total False Negative: {0}".format(total_fn))
			print("Total Sensitivity (TPR): {0}".format(sensitivity))
			print("Total Specificity (TNR): {0}".format(specificity))
			print("Total Precision: {0}".format(precision))

			# [entropy_list, zscore_list] = model.get_zscores(sess, chars, vocab,
			# 	token_file, entropy_dist_stats)
			# display_results(token_file, entropy_list, zscore_list, entropy_dist_stats)

def str2bool(s):
	return s.lower() in ('t', 'true', '1', 'yes')

# Add start and end token here?
def get_tokens(source_file):
	with open(source_file, 'r') as f:
		tokens = f.read().split()
	return tokens

def get_token_types(tokens, start_token, end_token):
	lexer = get_lexer_by_name('C')
	token_types = []*len(tokens)
	for token in tokens:
		token_types.append(get_token_type(lexer, token, start_token, end_token))
	return token_types

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

def get_token_outliers(entropy_scores):
	outliers = [None]*len(entropy_scores)
	for i in range(len(entropy_scores)):
		outliers[i] = is_outlier(entropy_scores[i])
	return outliers

def is_outlier(entropy):
	return entropy > 16

def get_line_outliers(entropy_scores, buggy_file):
	outliers = []
	start = 0
	end = 0
	with open(buggy_file, 'r') as f:
		for line in f:
			tokens = line.split()

			if len(tokens) == 0:
				continue

			end = start + len(tokens)
			line_entropy_scores = entropy_scores[start:end]
			outliers.append(is_outlier(line_summary(line_entropy_scores)))
			start = end
	return outliers

def line_summary(entropy_scores):
	# return np.mean(entropy_scores)
	return np.max(entropy_scores)


def get_outliers_by_type(entropy_scores, token_types, entropy_dist_stats):
	outliers = [None]*len(entropy_scores)
	for i in range(len(entropy_scores)):
		outliers[i] = entropy_dist_stats[token_types[i]].is_outlier(entropy_scores[i])
	return outliers

def display_results(buggy_tokens, fixed_tokens, buggy_outliers, changed):
	for i in range(len(buggy_tokens)):
		if i >= len(fixed_tokens):
			fix_token = ''
		else:
			fix_token = fixed_tokens[i]
		print("{0:<10} -> {1:<10} Change: {2:<6} Outlier: {3}".format(buggy_tokens[i], fix_token, changed[i], buggy_outliers[i]))

# def display_results(token_file, entropy_list, zscore_list, entropy_dist_stats):
# 	print("Token                Entropy    Z-Score     Outlier")
# 	print("====================================================")
# 	with open(token_file, 'r') as f:
# 		token = f.readline()[:-1]
# 		for i in range(len(entropy_list)):
# 			token = f.readline()[:-1]
# 			entropy = entropy_list[i]
# 			zscore = zscore_list[i]
# 			is_outlier = entropy_dist_stats.is_outlier(entropy)
# 			print("{0:<20} {1:<.4f}     {2:<3.4f}      {3}".format(token, entropy, zscore, is_outlier))


if __name__ == '__main__':
	main()

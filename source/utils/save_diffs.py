import argparse
import csv
import os
from subprocess import Popen, PIPE
from six.moves import cPickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--b_f_csv', type=str, default='../../data/student/buggy_fix_ce_list_updated.csv',
					   help='csv specifying buggy fix pairs')
	parser.add_argument('--granularity', type=str, default='line',
					   help='granularity to check diff. Use line or token.')

	args = parser.parse_args()
	validate_args(args)
	save_diffs(args.b_f_csv, args.granularity)

def validate_args(args):
	assert (args.granularity == 'line' or args.granularity == 'token'), 'Granularity must be token or line'
	assert os.path.isfile(args.b_f_csv), "csv file {0} doesn't exist".format(args.b_f_csv)

def save_diffs(b_f_csv, granularity):
	token_level = (granularity == 'token')
	if token_level:
		diff_ext = '.diff_token.pkl'
	else:
		diff_ext = '.diff_line.pkl'

	with open(b_f_csv, 'r') as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None) # Skip header line
		i = 0
		for row in reader:
			buggy_file = row[0]
			fixed_file = row[1]

			with open(buggy_file, 'r') as f:
				buggy_tokens = f.read().split()
			with open(fixed_file, 'r') as f:
				fixed_tokens = f.read().split()

			if token_level:
				# Will now need to line separate if want token level
				buggy_file = '/tmp/buggy.txt'
				fixed_file = '/tmp/fixed.txt'
				create_line_seperated_file(buggy_tokens, buggy_file)
				create_line_seperated_file(fixed_tokens, fixed_file)

			pipe = Popen(['diff', buggy_file, fixed_file], stdout=PIPE)
			diff_output = pipe.communicate()[0].decode('utf-8')

			if token_level:
				num_lines = len(buggy_tokens)
			else:
				num_lines = file_len(buggy_file)

			modified_list = get_modified_list(diff_output, num_lines)
			diff_file = row[0] + diff_ext
			with open(diff_file, 'wb') as f:
				cPickle.dump(modified_list, f)

def create_line_seperated_file(tokens, dest):
	with open(dest, 'w') as out_f:
		for token in tokens:
			out_f.write(token + "\n")

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_modified_list(diff_output, num_lines):
	modified_list = [False]*num_lines
	lines_changed, lines_added = get_diff_lines(diff_output)

	for line_num in lines_changed:
		# Direct line change
		modified_list[line_num-1] = True

	for line_num in lines_added:
		# Line above changed
		if line_num >= num_lines:
			modified_list[line_num-1] = True
		else:
			modified_list[line_num] = True

	return modified_list

def get_diff_lines(diff_output):
	lines_changed = []
	lines_added = []
	for line in diff_output.split("\n"):
		if len(line) > 0 and not line[0].isdigit():
			continue

		curr = ''
		is_add_op = ('a' in line)
		reading_range = False
		for char in line:
			if char == 'c' or char == 'd' or char == 'a':
				if len(curr) > 0:
					if not reading_range:
						start = int(curr)
					end = int(curr)
					if is_add_op:
						for i in range(start,end+1):
							lines_added.append(i)
					else:
						for i in range(start,end+1):
							lines_changed.append(i)
				break
			elif char == ',':
				if len(curr) > 0:
					reading_range = True
					start = int(curr)
				curr = ''
			elif char.isdigit():
				curr += char

	return lines_changed, lines_added

if __name__ == '__main__':
	main()
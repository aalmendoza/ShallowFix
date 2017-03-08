class DistributionStats:
	def __init__(self, mean, sd, q1, q2, q3):
		self.mean = mean
		self.sd = sd
		self.q1 = q1
		self.q2 = q2
		self.q3 = q3

	def is_outlier(self, x):
		multiplier = 3
		iqr = self.q3 - self.q1
		lower_fence = self.q1 - (iqr * multiplier)
		upper_fence = self.q3 + (iqr * multiplier)
		return (x <= lower_fence or x >= upper_fence)

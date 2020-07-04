class Scheduler(object):
	def __init__(self,queue,next_scheduler = None):
		self.next_scheduler = next_scheduler
		self.queue = queue # where the scans are sent
	def set_next(next_scheduler):
		self.next_scheduler = next_scheduler
	def schedule(self,last_scan):
		# throw not implemented

class TopN(Scheduler):
	def __init__(self,queue,next_scheduler = None,N = 10):
		# call super constructer...can never remember how to do this in python
		self.N = N
	def schedule(self,last_scan):
		# make the N scans
		for scan in  new_scans:
			self.queue.push(scan)

		# make an MS1
		ms1_scan = ....
		self.queue.push(ms1_scan)

		return ms1_scan.id_number,self.next_scheduler

class Just_ms1(Scheduler):
	def __init__(self,queue,next_scheduler):
		# call super constructor

	def schedule(self,last_scan):
		# make an ms1
		ms1_scan = ...
		self.queue.push(ms1_scan)
		return ms1_scan.id_number,self.next_scheduler

def main():
	# example of use
	topN = TopN(queue,N = 10)
	topN.set_next(topN) # always calls itself
	just_ms1 = Just_ms1(queue,next_scheduler = topN)

	final_id,next_scheduler = just_ms1.schedule(None) # no last scan


	all_scans = []
	while True:
		scan = next_scan() # wait for the next one
		all_scans.append(scan)
		if scan == None:
			break # finished
		while not scan.id_number == final_id:
			scan = next_scan()
			all_scans.append(scan)
		final_id,next_scheduler = next_scheduler.schedule(scan)

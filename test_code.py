import numpy as np

len_pred_seq = 6
end_ix = 0
#seq = np.arange(0, 15888, 1)
seq = np.arange(0, 530, 1)

len_seq = len(seq)

print(len_seq)
Y = []
#print(seq)
for x in range(0, len_seq, len_pred_seq):
	cols = []
	end_ix = x + len_pred_seq
	#print(x, end_ix)
	pred_seq = seq[x:end_ix]

	#print('matrix: ', pred_seq)
			
	pred_seq = np.matrix(pred_seq).flatten().tolist()			
	#if x >=15850:
	print(pred_seq)	
	Y.append(pred_seq)


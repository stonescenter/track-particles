from scipy.spatial import distance
from multiprocessing import Pool
import random

processes = 5

points =[(random.randint(0,10), random.randint(0, 10)) for i in range(100)]

# Group data into pairs in order to compute distance
pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
#print(pairs)
len_data = len(pairs)
chunk_size = int(len_data/processes)
print(len_data)
print(chunk_size)

# Split data into chunks
l = [pairs[i:i+chunk_size] for i in range(0, len_data, chunk_size )]
print(l)
def worker(lst):
    return [distance.euclidean(i[0], i[1]) for i in lst]

if __name__ == "__main__":
    p = Pool(processes)
    result = p.map(worker, l)
    # Flatten list
    #print([item for sublist in result for item in sublist])


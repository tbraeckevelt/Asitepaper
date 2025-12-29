import numpy as np
for i in range(8):
    num_ranks = 2**i
    for j in range(int(128/num_ranks)):
        f = open("myrankfile_"+str(num_ranks)+"_"+str(j), "w")
        for k in range(num_ranks):
            slot_a = int(np.floor((num_ranks*j +k) / 64))
            slot_b = int((num_ranks*j +k) % 64)
            f.write("rank "+str(k)+"=localhost slot="+str(slot_a)+":"+str(slot_b)+"\n")
        f.close()

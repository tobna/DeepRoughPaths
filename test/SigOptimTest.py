from RDEs import *
from test.test_utils import getsize
from time import time
import iisignature

batch_size = 1
n = 1

# n=5 => meins 30s iisig 90s

start = time()
B_hat = StratonovichBrownianRoughPath(n, batch_size)

B_hat.sig(1., 10, delta_t_max=0.001)
end = time()
my_sig = [entry.item() for entry in B_hat.sig(1., 10)]
print(my_sig)

print(f"duration\t= {end-start:.2f}s")
print(f"getsize(B_hat)\t= {getsize(B_hat)}")

iipath = cat([B_hat(t)[0].view(1, -1) for t in sorted(B_hat.vals_bm1.keys())], dim=0).numpy()

start = time()
sig = iisignature.sig(iipath, 10)
end = time()
print(list(sig))
print(f"iisig duration\t= {end - start:.2f}s")

for i, sigs in enumerate(zip(my_sig, [1.] + list(sig))):
    my_s, s = sigs
    print(f"Level {i}:\trel diff={my_s - s}")

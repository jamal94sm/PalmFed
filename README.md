palmfl: 
Train and test IDs are the same.
IDs are shared among clients
each client holds one domain (spectrum)
k = 20% (adjustable) of samples in each IDs of each client are allocated to test set
samples in test set are split into gallery and probe sets by a 50% split


palmfl_v2: 
Train and test IDs are different.
train IDs are shared among clients
each client holds one domain (spectrum)
k = 20% (adjustable) of IDs are allocated to test set
samples of test IDs in test set are split into gallery and probe sets by a m = 50% (adjustable) split


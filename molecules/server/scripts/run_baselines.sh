# NOTE: you might need to first restart database and remove solution_ folders.

#for protein in GSK3β_server DRD2 JNK3; do
for protein in DRD2; do
  PYTHONPATH=`pwd` PORT=8000 python solutions/run.py -b 1000 -w ml -t $protein -u baseline-ml
  PYTHONPATH=`pwd` PORT=8000 python solutions/run.py -b 1000 -w mutate -t $protein -u baseline-mutate
  PYTHONPATH=`pwd` PORT=8000 python solutions/run.py -b 1000 -w mutate_low_exp -t $protein -u baseline-mutate-low-exp
  PYTHONPATH=`pwd` PORT=8000 python solutions/run.py -b 1000 -w random -t $protein -u baseline-random
done
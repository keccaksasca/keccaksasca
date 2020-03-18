Software accompanying the paper "Single-Trace Attacks on Keccak" which will be published in the [IACR Transactions on Cryptographic Hardware and Embedded Systems](https://tches.iacr.org/) (TCHES) 2020 Issue 3.

Authors:
  - [Matthias J. Kannwischer](https://kannwischer.eu/) `<matthias@kannwischer.eu>`
  - [Peter Pessl](https://www.iaik.tugraz.at/person/peter-pessl/) `<peter.pessl@iaik.tugraz.at>`
  - [Robert Primas](https://www.iaik.tugraz.at/person/robert-primas/) `<rprimas@gmail.com>`

The paper is available at: [TODO] 


All simulation results were performed using the provided code.

Installation of required packages: `python3 -m pip install -r requirements.txt --user`

Performing simulations:
Change to directory `keccak-p/ `and run `python3 bp.py`. 
Alternatively, call `./inst_generic_xxx_xxx.sh` for one of the predefined scenarios.

Important command line parameters for simulation:
`-l`: lane length. default = 64 (keccak-f[1600])
`-s`: sigma for noisy Hamming weight. default = 1.0
`-w`: processor wordsize. default = 8
`-c`: clustersize. default = 8. Supported are 1, 8, 16
`-r`: number of simulated keccak-f rounds.  default = 2
`-d`: damping factor alpha. default = 0.75
`-i`: maximum number of belief-propagation iterations. default = 50

`--istate`: input state as 1600bit hexstring.
`--imask`: separate input bits to known and unknown parts. 1600bit hexstring. 1 = known bit, 0 = unknown bit
`--fmask`: separate output bits to known and unknown parts. 1600bit hexstring. 1 = known bit, 0 = unknown bit. default all zero, due to typically simulating only a small number of rounds.
Shortcuts for above parameters: R = fully random, 0 = all zero, supports python string expansion, e.g., `"RR"*16 + "00"*(200-16)` sets first 16 bytes random, others to zero

`--seed`: allows specifying a randomness seed (32bit hexstring) for rerunning specific simulations

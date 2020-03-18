#!/bin/bash
#
# SCHEME
#
# Name (name)
# Lane length (lanelength)          : Length of one lane of the keccak state in bits
# Permutation Rounds (rounds)
#
# PRIOR KNOWLEDGE
#
# Initial state (istate)            : Hex string + 'R' that indicates random bits
# Known initial mask (imask)        : Hex string that defines bit locations with known values
# Known final mask (imask)          : Hex string that defines bit locations with known values
#
# LEAKAGE
#
# Word size (wordsize)              : Data path with of the processor
# Noise (sigma)                     : The noise in the simulated leakage, 0 < sigma < ∞
#
# BELIEF PROPAGATION
#
# Iterations (iterations)
# Message damping (damping) [OPT]   : messages loose weight over time, forces convergence, disabled when 1
#
# PLOTS
#
# Make plots (blocking) [OPT]

python3 -u bp.py \
--name "keccak-p1600-generic-128-rand" \
--lanelength 64 \
--rounds 2 \
--istate R \
--imask '"00"*16 + "FF"*(200-16)' \
--fmask 0 \
$@

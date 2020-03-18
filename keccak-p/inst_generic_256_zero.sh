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
--name "keccak-p1600-generic-256-zero" \
--lanelength 64 \
--rounds 2 \
--istate '"RR"*32 + "00"*(200-32)' \
--imask '"00"*32 + "FF"*(200-32)' \
--fmask 0 \
$@

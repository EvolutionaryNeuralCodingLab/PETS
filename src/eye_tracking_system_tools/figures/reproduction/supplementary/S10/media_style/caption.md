Figure S10. Frame-to-frame tracking noise in quiet intervals is close to Rayleigh.

Histograms of D = hypot(Δφ, Δθ) in GUI-audited stationary windows, with a Rayleigh
overlay scaled to the sample median (B = median(D)/√(2 ln 2)). The dashed orange
line marks median(D), the typical noise magnitude. Where a detector threshold is
defined it is shown; SNR = threshold / median(D).

**(a)** Lizard (PV_228, block_016; n = 5640). median(D) = 0.113°/frame, B = 0.096°/frame,
threshold = 0.8°/frame, SNR = 7.05. KS D = 0.057.

**(b)** Mouse (M_002, block_012; n = 5204). median(D) = 0.305°/frame, B = 0.259°/frame,
threshold = 3.23°/frame, SNR = 10.58. KS D = 0.071.

**(c)** Turtle (T_18, block_001; n = 2692). median(D) = 0.163°/frame, B = 0.138°/frame,
threshold = 2°/frame, SNR = 12.28. KS D = 0.106.

The body of each histogram follows the median-matched Rayleigh overlay. KS D is the
largest vertical gap between the empirical and Rayleigh CDFs. With several thousand
samples those gaps exceed the Lilliefors 5% critical D, so a formal test rejects
*exact* Rayleigh; the residual is a modest heavy tail rather than a different
family.

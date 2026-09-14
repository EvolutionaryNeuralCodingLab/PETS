Figure S10. Frame-to-frame tracking noise in quiet intervals is close to Rayleigh.

Histograms of D = hypot(Δφ, Δθ) in GUI-audited stationary windows, with a Rayleigh overlay
scaled to the sample median (B = median(D)/√(2 ln 2)). The dashed 2B line is the 2 axis-σ
noise radius. Where a detector threshold is defined it is shown; SNR = threshold / 2B.

**(a)** Lizard (PV_228, block_016; n = 5640). median(D) = 0.113°/frame,
B = 0.096, 2B = 0.193,
threshold = 0.8°/frame, SNR = 4.15. KS D = 0.057.

**(b)** Mouse (M_002, block_012; n = 5204). median(D) = 0.305°/frame,
B = 0.259, 2B = 0.519,
threshold = 3.23°/frame, SNR = 6.23. KS D = 0.071.

**(c)** Turtle (T_18, block_001; n = 2692). median(D) = 0.163°/frame,
B = 0.138, 2B = 0.277.
No detection threshold. KS D = 0.106.

The body of each histogram follows the median-matched Rayleigh overlay. KS D is the largest
vertical gap between the empirical and Rayleigh CDFs. With several thousand samples those
gaps exceed the Lilliefors 5% critical D, so a formal test rejects *exact* Rayleigh; the
residual is a modest heavy tail rather than a different family. Lizard and mouse are closest
(D = 0.057 and 0.071); turtle has a thicker tail (D = 0.106).

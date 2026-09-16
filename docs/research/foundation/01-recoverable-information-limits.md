# Recoverable Information Limits in Real Video

## 1. Executive assessment  
Multi-frame approaches **can** recover true spatial detail beyond a single decoded frame, but only under stringent conditions.  In theory (Papoulis’ Generalized Sampling theorem) a bandlimited continuous signal can be exactly recovered from multiple sampled and differently filtered images.  Practically, this requires that individual frames be *undercaptured* (aliased) yet differently sampled – e.g. by subpixel shifts – so that their combined observations satisfy the sampling theorem.  Only in those cases do frames carry complementary frequency components.  If these conditions hold (aliased input, diverse shifts/PSFs), multi-frame reconstruction can extend the effective cutoff beyond that of one frame.  

However, real video pipelines often **violate** these conditions.  Typical consumer cameras use lenses and sensors that blur or filter high frequencies, and color subsampling (Bayer, 4:2:0, etc.) that destroys chroma detail.  Lossy compression quantizes away many high-frequency DCT coefficients in every frame.  Motion blur can nullify particular frequencies unless varied across frames.  In practice, state-of-the-art burst-SR experiments (e.g. Google Pixel) show **significant gains only up to modest zoom (~1.5–2×)**; beyond that, additional frames give little or no extra detail.  At low SNR or with extreme blur, even subpixel diversity yields minimal recovery.  Conversely, if a scene is static and frames are identical (aside from noise), no algorithm can invent new high frequencies.  

**Implications:**  Temporal reconstruction is promising only in regimes with real sampling diversity (hand tremor, jitter, controlled motion, mixed blur).  In optimal cases (sharp optics, meaningful jitter), it can effectively act like a “super-resolution camera” and recover detail up to the optical cutoff.  In degraded regimes (static scenes, heavy blur, oversampling, severe quant), it offers at best denoising or interpolation.  The architecture should explicitly detect these regimes: pursue multi-frame fusion when theoretical sampling gains exist, and fallback to single-frame methods otherwise.  Key takeaways are summarized below (with supporting evidence):

- **Aliasing is essential.** If frames are *already* Nyquist-sampled (blurring or OLPF preventing alias), then multiple samples provide no new frequency support (documented by sampling theory). Conversely, if aliasing exists, subpixel offsets can disambiguate it.  
- **Shift diversity matters.** Only certain pixel displacements yield independent information.  Uniform coverage of a finer grid (e.g. 2×2 offsets for 2× upscaling) is optimal.  In practice, random hand tremor suffices to approximate uniform subpixel shifts.  
- **Gains saturate quickly.** Empirical studies show large improvements at modest zoom (∼1.5×) but diminishing returns beyond ~2×.  Each extra frame adds less new information as the reconstruction approaches the alias-inversion limit.  
- **Blur null-filling can help.** If frames have *different* blur kernels (e.g. varied exposure or motion blur), they can complement each other.  MERL’s “PSF null-filling” result proves combining multiple non-invertible PSFs can yield an invertible overall blur.  In practice, this means changing exposure/time or focus between frames can unlock higher frequencies otherwise lost.  
- **Codec and CFA losses are permanent.**  Luma channels preserve spatial structure, but chroma subsampling (4:2:0/4:2:2) irreversibly halves or quarters color resolution.  Similarly, quantization in H.264/HEVC discards high-frequency DCT coefficients.  No temporal fusion can recover detail obliterated by these processes.  

**Judgment:**  The premise that *some* real video frames carry extra recoverable detail is well-supported (documented by sampling theory and practical burst-SR work).  However, it is **not universally strong**; its validity is highly conditional.  The architecture should therefore be adaptive: aggressively exploit multi-frame super-resolution in favorable cases (high SNR, subpixel motion, moderate aliasing) and avoid wasteful fusion when conditions fail.  The rest of this report details the theory and evidence behind these conclusions, delineates precise conditions, and recommends how to test for recoverable regimes.

## 2. Definitions and observability  
We define **recoverable detail** as genuine spatial content of the original scene that can be inferred *uniquely* from the decoded video frames under known physics.  We distinguish:

- **Directly recoverable detail:** Frequencies present in the scene that survive image formation (lens MTF, sensor aperture, sampling) and appear (possibly aliased) in the frames.  These are theoretically invertible by multi-frame fusion if sampled differently.  
- **Model-constrained recoverable detail:** Frequencies not fully measured but constrained by a known model (e.g. a blur kernel).  For example, if blur is known and invertible (no zeros), deconvolution yields higher frequencies.  If not fully invertible, only partial inversion is possible.  
- **Statistically inferable detail:** Features that cannot be deterministically resolved from observations but can be *estimated* given assumptions (e.g. sub-resolution text patterns if any faint hint appears).  This may be quantified by limits (Cramér–Rao bounds) but is not guaranteed detail.  
- **Prior-driven synthesis (not recoverable):** Detail introduced solely by priors or learning, without support in the data.  E.g. hallucinated texture in a face based on a generic face model.  This falls outside “recoverable” by our definition.  

We operationalize “recoverable detail” in terms of *identifiability*.  Given the forward image formation (optics, sampling, compression) and a set of frames, the recoverable band is the range of spatial frequencies for which the inverse problem is well-conditioned.  Frequencies lying in the **null space** of the measurement model (e.g. zeros of the combined optical+sensor+alias transfer function) are irretrievable.  Frequencies outside the null space but with low SNR might be theoretically present but practically lost (estimation variance too high).  In practice, we will consider detail recoverable if it can be reliably reconstructed (e.g. above some SNR threshold).  

*Observability* means what the decoded video actually contains.  Video frames are digital signals after color filtering, demosaicing, gamma, downsampling, and compression.  All analysis herein refers to detail observable *post-decoding*.  Metadata (motion vectors, depth maps, etc.) is excluded, as are side-channel data.  Thus “observation” = sequence of decoded RGB frames (possibly with 8-bit precision and chroma subsampling).  

## 3. Sampling theory  
An essential foundation is Nyquist sampling.  A single video frame, sampled at pixel pitch $T$, can only support spatial frequencies below $f_N=1/(2T)$.  Frequencies above $f_N$ are aliased (overlap) in that frame and lost.  Multiple frames with different sampling phases can disambiguate aliased components: a classic 1D example is two sampled signals offset by half a pixel, which together allow reconstruction up to twice the single-frame Nyquist frequency.  In 2D, offsetting on a subpixel grid (e.g. quarter-pixel shifts in X and Y) can cover a finer grid and capture higher bandlimit information.

**Mathematical conditions:**  For multiple samples to add information, their sampling patterns must be *non-uniform*.  Uniformly spaced frames with integer-pixel shifts carry identical samples (no new information).  The Generalized Sampling Theorem (Papoulis, 1977) shows that a band-limited continuous signal can be reconstructed from the samples of its convolutions with different known filters, provided the filters collectively have non-overlapping zeros.  In imaging terms, each frame provides a filtered (blurred) and subsampled observation of the scene.  If the blurs (or point spread functions) differ and their frequency nulls do not coincide, then combining frames can fill in the missing frequencies.  

Concretely, Tsai & Huang (1984) distilled two requirements for multi-frame SR: (1) **aliasing** – the LR images must have aliased high frequencies, and (2) **diverse shifts** – the LR images must be sampled at different subpixel offsets.  When these hold, each LR image carries a different “phase” of the original spectrum.  By the Fourier shift theorem, a shift of the scene causes the high frequencies to produce different low-frequency aliases in each frame.  Collectively solving those phase differences recovers the original spectrum.  

In more rigorous terms, the LR frames $y_k[m,n]$ relate to the high-res scene $x(x,y)$ by  
```
y_k = D B_k (M_k x) + noise, 
```  
where $M_k$ is motion (warp), $B_k$ is blur/PSF, $D$ is downsampling.  The effective multi-frame system matrix (mapping $x$ to all $y_k$) must have full column rank over the extended frequency band to recover $x$ uniquely.  If it does, the inverse problem is well-posed up to noise.  In practice, this rank condition translates to having enough independent equations (frames) covering each frequency.  As a rule of thumb, ideal uniform shifts at an *r×r* grid allow up to *r×* resolution improvement (since $r^2$ distinct samples cover the finer grid).  For example, perfect offsets of (0,0),(0,0.5),(0.5,0),(0.5,0.5) could in principle reconstruct a 2× upscaled image.  However, **quantitative bound** typically requires more frames under noise and blur.

**Subpixel displacement:**  Small shifts – even fractional pixels – can be highly effective.  Random hand tremor (~8–12 Hz, small amplitude) provides a pseudo-random jitter that approximates a dense subpixel sampling pattern.  Controlled camera shifts (the “jitter camera” concept) also show that deliberate micron-level motions yield tangible resolution gains.  Conversely, if shifts are collinear (e.g. all frames move horizontally only) or commensurate (e.g. always integer multiples), some frequency components may remain unsampled.  In general, any pixel-phase diversity helps; best is isotropic coverage.

**Gains vs frames:**  Each additional frame adds equations but also noise.  Theoretical analyses (e.g. CR bounds) predict diminishing returns: resolution improvement grows slowly with SNR and frame count (often a sub-linear power law).  Empirically, multi-frame algorithms often saturate quickly: e.g. Wronski et al. found marked gains at 1.5× zoom with 8–16 frames, modest further gains to 2×, and nothing beyond.  This suggests on typical camera data *only ~2× the LR band* is recoverable with few frames; going to 3× or 4× likely needs an impractically large number of frames or unrealistically clean conditions.  No well-known bound fixes exact numbers, so experiments (Sec. 10) are needed to quantify for specific pipelines.

## 4. Real image-formation pipeline  
Every pixel in a decoded video has undergone a chain of transformations that alter its information content.  We briefly trace these stages and note their effect on high-frequency detail:

- **Optics (Lens & Aperture):** The lens and aperture produce an optical PSF.  An ideal diffraction-limited lens of f-number *f* and wavelength λ has an optical cutoff frequency $f_c≈1/(λ f)$.  Above $f_c$, contrast is zero.  Aberrations, diffraction, and defocus reduce the *effective* cutoff below the theoretical maximum.  For example, defocus can sharply attenuate mid-range frequencies long before the diffraction limit.  In practice, many consumer lenses yield a limited MTF: i.e. even *in-focus* contrast may be low at higher frequencies.  **Effect:** highest frequencies may be attenuated or lost before ever reaching the sensor.

- **Sensor integration:** Each pixel integrates light over its finite area (box filter PSF) and over time (exposure).  The pixel aperture acts as a sinc low-pass in frequency.  A smaller pixel (higher resolution) passes more detail, but real pixels have non-negligible size and fill factor.  **Effect:** reduces MTF at high frequency.

- **Optical Low-Pass Filter (OLPF):** Some cameras include a (slight) blur filter to mitigate aliasing.  If present, it deliberately cuts off frequencies above half the sensor sampling frequency.  **Effect:** enforces Nyquist compliance at the cost of detail.

- **Color Filter Array (CFA) / Demosaicing:**  Most video cameras use a Bayer CFA, sampling only one color per 2×2 pixel block.  This halves green-channel resolution and quarters red/blue relative to the sensor grid.  In effect, the green channel is 50% more aliased and red/blue up to 2× more aliased.  Demosaicing algorithms interpolate missing colors but cannot recover frequencies not measured by the CFA.  **Effect:** color edges beyond 0.5× pixel sampling are irretrievably lost or aliased (unless multiple frames help, see Sec. 7).

- **In-Camera Processing (NR, Sharpening, etc.):**  Cameras may apply spatial denoising (smoothing), downscaling (digital zoom), or sharpening.  Denoising/anti-noise filters blur fine detail, degrading high frequencies.  Sharpening can boost perceived acutance but often boosts noise/artifacts rather than true detail.  **Effect:** noise reduction trades off with detail; any nonlinear or undocumented filtering complicates reversal.

- **Gamma and Tone Mapping:** Video frames are usually gamma-encoded (nonlinear).  Provided the gamma curve is known and not quantized away, linearization is possible.  However, aggressive tone mapping (especially in HDR workflows) can crush highlights/shadows and alter local contrast, effectively losing information in those regions.  **Effect:** nonlinear mapping can obscure true radiometric contrast; recoverability requires knowing and inverting the curve.

- **Digital Stabilization / Cropping:**  If camera stabilization is on, the camera may shift sensor frames to compensate for motion, often cropping the field of view.  This can remove any intentional jitter and reduce usable frames for SR.  **Effect:** reduces or scrambles subpixel motion diversity.

- **Downsampling / Digital Zoom:**  Some devices downsample the raw sensor readout (binning or digital zoom).  If frames have been upscaled or downscaled in camera, the original detail is blurred or aliased by the resampling kernel.  **Effect:** multiple rounds of resampling (if footage has been re-encoded) compound blur and aliasing losses. These losses are **irreversible** without priors.

- **Chroma Subsampling:**  In encoding, common formats (4:2:2, 4:2:0) drop horizontal/vertical chroma samples to reduce bitrate.  For example, 4:2:0 halves chroma resolution in both dimensions.  After decoding, chroma channels must be upsampled, but that only interpolates what was not captured.  **Effect:** no temporal processing can recover actual chroma detail beyond the reduced sample count.  Only luma is fully preserved (aside from transform quant losses).

- **Compression (Quantization):**  Video codecs (H.264/HEVC/VP9/AV1) transform blocks (DCT or wavelets) and quantize coefficients.  High-frequency coefficients are often quantized to zero at moderate bitrates.  The quantization process is essentially adding uniform “noise” to each residual.  Different frames may quantize differently if content changes, but the underlying lost information cannot be regained.  **Effect:** frequencies above the codec’s effective cutoff (dependent on QP and bit budget) are irretrievably discarded.  Some high-frequency “dithering” can occur, but it functions as noise and typically **does not help** resolve aliasing (it only reduces certainty).  

- **Deblocking/Sample-Adaptive Filters:**  After decoding, filters like HEVC’s SAO or VP9’s in-loop filter smooth edges.  They improve visual quality but slightly blur sharp transitions.  **Effect:** marginally reduces very high-frequency edge contrast but does not create new detail.

In summary, the pipeline typically *attenuates* or *destroys* high frequencies rather than preserving them.  Recoverable information is limited by the **narrowest bottleneck**: e.g. if the lens cutoff is 60 cycles/mm and the sensor Nyquist is 100 cycles/mm, the lens sets the limit.  In any stage where an irrecoverable loss (alias or zero) occurs, no amount of computation can restore that detail.  

## 5. Blur and PSF limits  
Optical blur (defocus or motion) acts as a low-pass filter.  Its null spaces define irrecoverable frequencies.  **Single-PSF limitation:** If *all* frames share the exact same blur PSF $h$ whose Fourier transform $H(f)$ has zeros, then any frequency where $H(f)=0$ is lost in every frame.  Multi-frame fusion cannot recover such “null-space” frequencies.  In other words, repeated identical blur gives no new information.

By contrast, **multiple varying PSFs can help**.  If different frames have different blur kernels with non-overlapping zeros, their combination can fill in missing frequency bands.  MERL’s “PSF null-filling” demonstrates this: combining multiple exposures of varying lengths yields a *jointly invertible* PSF whose frequency response is nonzero everywhere.  Concretely, they show that changing exposure time across frames makes the **combined OTF null-free**, enabling stable deconvolution.  In effect, low-pass filtering by one PSF is undone by another if their nulls differ.  Thus: **varying blur across frames can unlock detail** unavailable in any single image.

However, there are practical caveats.  PSF must be accurately known or estimated for inversion; uncertainty in blur estimation degrades recovery.  Blind deconvolution from video can leverage multiple frames to estimate blur (as [79] discusses), but real videos rarely record precise blur parameters.  Also, mixing very differently blurred frames (e.g. short and long exposure) can introduce severe noise trade-offs or artifacts if not handled carefully.  

**Summary:**  *Multiple different blurs can exceed any single frame’s detail* – this is a documented result.  Conversely, if all frames share the same PSF (e.g. same aperture and exposure), they share the same null-space and no temporal method can recover frequencies that were zeroed by the PSF (Established fact by theory).  In-camera blur (lens defocus, motion blur) therefore sets a hard boundary: beyond its cutoff, no reconstruction is possible without priors. 

## 6. Compression and resampling limits  
Video codecs irreversibly remove detail to reduce bitrate.  Key points:

- **Intra-frame transform quantization:**  Within each frame, the DCT (or other transform) coefficients above a certain rank are quantized coarsely or to zero.  Low-bitrate frames have **aggressive high-frequency suppression**.  This loss is mathematically irreversible – once a coefficient is zeroed, no amount of alternate sampling can bring it back.  

- **Inter-frame prediction:**  Predicted (P/B) frames rely on motion-compensated references.  In practice, a P-frame encodes only the residual difference from a predicted reference.  That means much of its content is shared with prior frames.  Intra-coded (I) frames carry the bulk of spatial detail.  One might hope that differences in which frame is intra-coded (or subtle differences in residuals) could reveal complementary detail.  However, if prediction is perfect, P/B frames carry essentially no new spatial frequencies.  In reality, imperfect prediction yields residuals, but those residuals are also quantized.  There is no guarantee (and little evidence) that “different DCT patterns” across frames systematically preserve extra high frequencies in complementary fashion.  Any differences are largely noise-like.  

- **Macroblock structure & filters:**  Codecs may smooth block boundaries or adjust quantization adaptively.  While clever coding can sometimes shift detail between frames, these effects are generally too subtle to leverage as independent measurements.  For example, Film-grain or dithering included in content are typically averaged out by codecs, not preserved.  

- **Repeated scaling/transcoding:**  If a video was scaled (e.g. from 1080p to 720p) and then re-encoded, the second encoding treats the input as its new “original”.  The first downscaling has already eliminated high frequencies.  Subsequent processing cannot restore them.  Transcoding from one lossy format to another compounds losses.  

In summary, **practical codecs irreversibly destroy high-frequency components**.  Any complementarity between frames is random (from noise or quantization dithering) and not a reliable source of signal.  We found no literature claiming that modern codecs preserve meaningful subpixel information across frames.  If anything, codec prefilters and quantization homogenize frames.  Therefore, the architectural focus should treat encoding loss as a one-way street: plan to recover detail only up to the coding cutoff, and disregard “frame-to-frame quantization differences” as a source of actual signal (they act like added noise).

## 7. Natural temporal diversity  
Real video may include many sources of frame-to-frame variation that serve as sampling diversity:

- **Hand tremor / camera shake:**  The most ubiquitous is involuntary camera jitter when handheld.  Studies show hand tremor is a small, quasi-random oscillation (~8–12 Hz) that moves the camera by fractions of a pixel each frame.  This provides subpixel offsets very close to ideal jitter.  Practical results confirm it: Pixel phone bursts rely on natural shake to drive SR.  If the camera is stabilized or on a tripod, micro-actuators can be used (Sony, Pentax, etc.), but otherwise hand shake usually suffices.

- **Controlled jitter (Jitter camera):**  Special camera designs “dither” the sensor by a known pattern.  The “jitter camera” prototype and follow-ups demonstrate that such dithering can significantly boost resolution for static scenes.  This is essentially the same idea as hand tremor but with certainty and without blur.  It validates that *even a mechanical dithering of <1 pixel* yields recoverable detail if blur is minimal.

- **Camera translation/rotation:**  Slow pans or nods introduce relative motion to static scenes.  If motion is known, this is equivalent to applying subpixel shifts across the frame (though large shifts complicate alignment).  Moderate smooth camera moves (intentional or from walking) can provide useful offset diversity, but alignment must handle scene parallax.  Large rotations can produce varying sub-pixel coverage in different parts of the image via projective geometry.

- **Object motion:**  Independent object movement (rigid or non-rigid) also moves features across the sensor.  In theory, each object can be treated as a “camera shift” on that part of the scene.  If objects move slowly relative to the frame rate, their motion might offer reconstruction clues.  However, occlusions and deformation break global alignment, so algorithms must detect and exclude inconsistent regions.

- **Parallax (camera translation with depth):**  When the camera moves laterally, scene points at different depths shift by different amounts.  This creates a range of effective subpixel offsets in a single shot (beyond what global motion alone does).  To exploit this, one must know or estimate depth for proper fusion.  While powerful (like multi-view stereo), it is beyond blind SR algorithms.

- **Rolling shutter:**  CMOS rolling shutter means each row is exposed at a slightly different time.  For moving scenes or jittering camera, this causes intra-frame spatial distortions.  In principle, this is another form of controlled sampling pattern (each row “sees” a slightly different version).  However, rolling-shutter warp is complex to invert and typically treated as distortion to correct, not an SR help.  It could in theory provide fine sub-row sampling, but only if accurately modeled (unknown in current decoders).

- **Vibration:**  High-frequency vibration (e.g. from a drone or vehicle) adds jitter but may be too rapid or rotational to help global alignment.  Controlled platform vibration (moving sensor) can be useful, but in consumer video this is usually unwanted noise.

**Expected distributions:**  Hand jitter on 60 Hz video often has a subpixel amplitude with roughly Gaussian distribution centered on zero.  Example data (Google) show one standard deviation ~0.9px diagonal, enough to uniformly cover the 2×2 subpixel grid.  Slow pans produce consistent linear displacements (not random), which can still sample details if subpixel.  Pure zoom or purely out-of-plane rotation typically change scale without adding new planar samples and are less useful for in-plane detail.

**Occlusions and misalignment:**  Moving objects and depth cause occlusions.  In areas where objects move differently from the background, multi-frame SR must treat them separately.  Typical burst pipelines mask out such regions.  Occluded areas lose the multi-frame advantage and revert to single-frame quality.  Thus real-world content with large occlusions (foliage, crowds) offers spatial diversity only on small patches; most regions will have only one good viewpoint.

**Summary:**  Natural handheld motion is generally beneficial: it provides subpixel offsets with minimal planning.  Slow global movements (pans) can also help if properly estimated.  Fast or complex motion (fast moving cars, wild camera swings) tends to violate assumptions and yields little extra detail.  Rolling shutter and parallax introduce subtle effects that might help if explicitly modeled, but are more often treated as nuisances.  We will evaluate these sources in Sec. 12 to decide which to pursue.  In general, **subpixel image-plane shifts (jitter/handshake) are the most straightforward and universal source of diversity**.

## 8. Static-video negative control  
Consider a video of a static scene with a fixed camera and *identical* frames aside from noise.  This is the worst-case (lack of diversity).  What can temporal processing do here?  **No new spatial information.**  All frames sample the same spectrum in the same way.  At best, multi-frame fusion can *denoise* the image (averaging out independent noise), but it cannot recover frequencies not present in any single frame.  If the codec adds independent noise-like artifacts (quantization noise), temporal averaging can slightly reduce that too, but again only by trading variance, not by adding resolution.

In more detail: if frames differ only by noise or compression jitter, then fusing them (e.g. by averaging) can improve SNR by $\sqrt{N}$.  The MTF remains that of one frame.  Multi-frame filters may reduce block artifacts or temporal flicker, but sharpening beyond the single-frame limit would be hallucination.  Indeed, stationary-PLNR (perceptual image processing) analyses show that, absent alias or motion, super-resolution techniques do nothing more than denoise.  Stated formally, the multi-frame system matrix becomes redundant (columns identical) and rank-deficient; invertible reconstruction is impossible.

**Experimental validation:**  We propose (Sec. 10) to use this static case as a “negative control.”  Synthetic tests should show that as motion → 0, reconstructed detail → single-image interpolation.  Any algorithm claiming to “sharpen” a static video beyond conventional upscaling must be injecting prior content rather than recovering real information.  Indeed, prior work on supervised video SR often tests static scenes and achieves only artifact suppression or detail invention, not true recovery.  This scenario is well-understood (Established fact) and requires no new evidence: absent sampling diversity, the only valid multi-frame benefit is noise reduction.

## 9. Quantitative recoverability bounds  
Where possible, we provide ballpark figures and recommend measurements:

- **Distinct subpixel samples needed:**  The *ideal* minimum for an $r\times$ upscale (isotropic) is roughly $r^2$ distinct phases (e.g. a uniform $r\times r$ offset grid).  For $r=2$, that suggests 4 well-chosen frames.  In practice, partial grids or irregular sampling can approach this with fewer frames if shifts are optimal.  Likewise, $3\times$ theoretically needs 9.  However, with real blur and noise, **much more than the ideal number may be needed** to approach performance.  We did not find literature giving exact counts under non-ideal conditions.  *Proposed test:* simulate an ideal PSF and noise-free downsampling, and verify the minimal frames for exact reconstruction.  

- **Displacement distribution for useful sampling:**  Best is roughly uniform random in [0,1) pixel in both axes, avoiding alignment on a coarser grid.  A known result is that any two frames with a rational offset whose components have denominator $q$ can at most reconstruct $q\times$ zoom.  In practice, subpixel offsets of about half a pixel in each direction are highly beneficial (e.g. see Gaussian jitter of ~0.5px RMS).  We did not find a strict “goldilocks” number, but empirically, jitter often spans ~[-1,1] px.  *Proposed test:* sweep random jitter distributions and measure how well the alias inversion condition holds.

- **Frame count effect:**  Gains diminish.  One can measure, for a fixed scenario, the PSNR (or high-frequency content) vs $N_{\rm frames}$.  Past work shows a rapid initial jump and plateau.  Quantitatively: Pixel3 results suggest significant improvement by ~10-16 frames for ~1.5×, with almost no gain beyond ~20.  We expect an asymptote as $N\to\infty$ determined by noise.  *Proposed test:* vary $N$ in controlled synthetic bursts and plot SF reconstruction vs $N$.

- **Alignment error:**  Misregistration acts like blur.  If residual misalignment is $\epsilon$ pixels, recovered detail above frequency $\sim 1/(2\epsilon)$ will be attenuated.  For example, 0.25px error roughly caps at 2× Nyquist of the subgrid.  We did not find explicit studies, but alignment precision is known to be critical.  *Rule:* if misalignment >0.3px, multi-SR fails (engineering inference).  *Proposed test:* corrupt registrations by known jitter and see when SR breaks.

- **Blur kernel width:**  Wider blur (larger PSF) narrows recoverable band.  For a Gaussian blur of width $\sigma$, the cutoff is ~$1/(2\pi\sigma)$.  A rough guideline: if blur $\sigma>1$px, frequencies above ~0.2Nyquist are suppressed.  Multi-frame cannot exceed the *joint* optical cutoff.  The null-filling result suggests if blur changes per frame (e.g. mixed $\sigma_1$ and $\sigma_2$) one might recover up to the larger-support (smaller blur) cutoff.  But if all $\sigma$ equal, the cutoff stands.  *Proposed test:* simulate bursts with varying known $\sigma$ and check highest recovered freq (e.g. MTF50).

- **Codec quantization:**  Hard to quantify without specifics.  Theoretically, a high QP yields heavy high-frequency loss.  One could define the “useful resolution” of a codec as the highest frequency whose average DCT magnitude is nonzero.  Empirical check: for common bitrates (e.g. 8 Mbps 1080p), we estimate luma cutoff ~0.5–0.8× Nyquist.  Temporal fusion won’t increase this cutoff.  *Proposed test:* encode a known high-frequency pattern and see which frequencies survive in I/P frames.

- **Max recoverable band:**  Bounded by the smallest of (optical cutoff, sensor Nyquist) for luma, and by 0.5× Nyquist (or lower) for chroma.  If multi-frame adds diversity, it can at best reach the optical cutoff.  For example, a lens of f/2 at 550 nm has ~112 cycles/mm cutoff; a 12 µm pixel has 42.5 cycles/mm Nyquist.  Here, frames could in principle extend to 112 c/mm if shifts sample those frequencies.  Without specific literature values, we conclude: **cannot exceed optical cutoff**.  

In most cases, literature offers qualitative or very scenario-specific numbers (e.g. [74] for a phone).  We advise careful experimentation with real and synthetic bursts to pin down these bounds for each target camera and codec setting.

## 10. Experimental protocols  
We propose a structured evaluation using high-quality ground truth:

- **A. Synthetic sampling tests:**  Take a known high-res image (e.g. resolution chart or natural scene).  Apply controlled subpixel shifts (rational offsets), then apply a fixed PSF (e.g. Dirac or mild blur) and downsample.  This yields multiple LR frames with exact ground truth.  Vary $N$, offset distributions (uniform grid vs random), and noise.  Metric: ability to recover the original high-res Fourier spectrum.  *Control:* use ideal interpolation (sinc) on one frame as baseline.

- **B. PSF/blur sweep:**  Use the above but vary the PSF width.  For each blur ($\sigma$ or defocus amount), generate LR frames (with or without offsets) and attempt SR.  Evaluate maximum recoverable frequency (e.g. MTF curve of output) vs single-frame.  *Control:* same frames with identical blur.  This isolates effect of PSF size on recoverability.

- **C. Motion-blur diversity:**  Simulate linear motion of scene during exposure.  Render multiple exposures of differing length or direction.  Attempt multi-frame deblurring and SR.  Metric: how much detail recovers beyond the sharpest single frame.  Check PSF null-filling: for a moving object, combine frames with varying exposure as in [79].

- **D. Codec sweep:**  Encode a fixed LR dataset (from A–C) at multiple codecs (H.264, HEVC, AV1, possibly VVC/VP9) and bitrates.  Also vary GOP structure (all-I vs low-delay P/B).  Then decode and run the same fusion tests.  Evaluate how much aliasing content remains.  *Key:* measure which frequencies survive in each frame and whether multi-frame fusion achieves higher effective MTF than single-frame for the same bitrate.

- **E. Natural-motion simulation:**  Create synthetic video of a 3D scene (e.g. from a game engine or renderer) where the camera undergoes controlled motion (translation, rotation, zoom) with known parameters.  Optionally include independent moving objects.  Downsample with optional simulated PSF/CFA, then test SR reconstruction with ideal or estimated motion.  This bridges from ideal to real complexity.

- **F. Real-camera validation:**  Acquire paired HR–LR data.  Approaches include: (i) optical setup with a beam splitter or an optical downsampling lens to capture HR and LR simultaneously; (ii) use a very high-res sensor and synthetically bin it to LR while capturing bursts; (iii) known high-res charts and a tripod so that any movement is only subpixel vibrations.  This tests the realism of the synthetic models.  *Ground truth:* The high-res video or images.  Metric: MTF, PSNR, subjective clarity.

Each experiment must include quantitative metrics: spectral response (MTF50), PSNR/SSIM on edges, and reconstructed spectrum plots.  Statistical variance (confidence intervals) should be reported over multiple trials.  We should also document failure cases (e.g. misalignment thresholds).  Frequency-domain plots of LR vs reconstructed images will highlight actual gains in recoverable band.  The “falsifying outcomes” include: if multi-frame results never exceed single-frame limits under expected diversity, the underlying premise is questionable.

## 11. Technology/tooling landscape  
Several existing tools and libraries can support this investigation:

- **Super-resolution toolkits:**  The FAU Multi-Frame SR Toolbox (MATLAB/C++) implements many classical algorithms (MAP, robust estimators) that can be used as references or baselines.  We can also adapt open-source implementations of shift-and-add or iterative back-projection.  *Adopt/Adapt:* Use these for prototyping fusion algorithms and verifying theory.  

- **Alignment and warping:**  Accurate subpixel registration is critical.  OpenCV (using pyrLK optical flow) or MATLAB’s imreg can align frames.  Google’s Pixel pipeline algorithm (Hasinoff et al.) and its implementations (e.g. OpenCV or research repos) provide robust multi-scale alignment.  *Adopt:* As a starting point for frame-to-frame registration.

- **Fourier-domain methods:**  Scripts to analyze and combine Fourier spectra of frames will help test sampling theory (e.g. repliations of [76] ideal sampling).  FFT libraries (numpy, MATLAB) suffice.  

- **Nonuniform resampling:**  Libraries like SciPy’s `interpolate` or custom sinc-resampling code can simulate arbitrary offsets.  This supports synthetic experiments (Sec 10A–E).  *Adopt:* For generating irregularly sampled LR images from HR ground truth.

- **PSF estimation/deconvolution:**  Blind deconvolution toolkits (e.g. MATLAB’s `deconvblind`, Python’s skimage.restoration) can test deblurring limits.  Algorithms for motion deblurring (e.g. Cornell’s code, Fergus et al.) might be adapted.  *Reference:* Use for analyzing potential blind inversion.

- **Video codecs:**  Use FFmpeg or reference codec software (x264, x265/HM, libvpx, SVT-AV1) to produce controlled encoded streams.  Tools like `btcdiff` (for bitstreams) or custom code can extract DCT coefficients to see which survive.  *Adopt:* FFmpeg for encoding, and possibly decode to raw frames for analysis.

- **Metric tools:**  Imatest or VIQ libraries can compute MTF from test charts.  Python libraries or custom code can measure energy spectra.  *Adopt:* Use standard image quality metrics (PSNR, SSIM) and add frequency-domain metrics.

- **Data sets:**  Existing high-speed or burst photography datasets (like Zurich RAW to RGB) may provide useful real-world sequences.  These can supplement synthetic tests.  

- **Hardware:**  Camera APIs (Android Camera2, DSLR SDKs) allow capturing raw bursts.  A motion stage (e.g. piezo platform) could introduce controlled shifts.  

Classification (adopt/adapt/use as reference): classical analytical methods and Fourier approaches (Papoulis theory, multi-image alignment) should be directly used or re-implemented.  State-of-the-art learned methods (neural SR networks, vision transformers) will *not* be used in experiments, but can be referenced for failure modes.  Codecs and PSF calibration tools are *adopt directly*.  Custom simulation (sampling, blur, motion) we must implement.  The FAU toolbox is *use as reference/baseline*.  

All tools chosen must be vendor-neutral and extensible.  For example, using FFmpeg and open libraries ensures compatibility across codecs and avoids reliance on GPU-specific APIs at this stage.

## 12. Implications for architecture  
Based on the above analysis, we answer the core decision points:

1. **Is the core premise valid?**  *Qualified yes.*  Theory and practice show that multi-frame fusion can recover real detail **when conditions are met**.  However, those conditions are stringent.  We should not take temporal diversity as a given but treat it as an **opportunity** to exploit when detected.

2. **Conditions where premise is strongest:**  High optical quality (diffraction-limited, no extra OLPF), sampling just below Nyquist (so aliasing exists), dynamic jitter or multiple vantage points, decent SNR, and moderate codec bitrate.  In particular, hand-held bursts with natural tremor, static or slowly-moving scenes, and carefully metered exposure (avoiding clipping) represent the sweet spot.

3. **Conditions where it’s weak/absent:**  - Any fully static (no subpixel motion) capture.  - Scenes out of focus (blurry) beyond recoverability, especially if blur is uniform.  - Fast, complex motion or severe occlusion, which break alignment.  - Very low light (low SNR), where noise dominates any high-freq signal.  - Ultra-high zoom (≥4×) where insufficient frames are available to sample needed phases.  - Extremely lossy compression (very high QP) or heavy chroma subsampling, which remove frequencies outright.  

4. **Diversity to explicitly seek:**  - *Subpixel shifts:* We should exploit any jitter or deliberate micro-motion.  If stabilizer is active, consider disabling it or using sensor-shift (OIS) intentionally to add diversity.  - *Varying blur:* Modes like HDR (multi-exposure) or AEB (auto exposure bracketing) can naturally vary blur and exposure.  Using bursts at different (even slightly) focus or aperture may help.  - *Multiple viewpoints:* If applicable (e.g. multiple cameras or multi-view video), leverage parallax.  - *Nonuniform sampling patterns:* If algorithmically possible, introduce controlled jitter (e.g. vibrating a lens).  

5. **Myths or prior-driven detail:**  - **“Any camera motion is good”:** Not true. Uniform integer translations do nothing, and excessive motion blur can hurt (as the Jitter Camera paper proved). Only subpixel jitter without blur, or varied blur, is useful.  
   - **“Temporal denoising = super-resolution”:** Denoising reduces variance but doesn’t extend frequency support.  Gains from denoising should not be conflated with true resolution recovery.  
   - **“Codec gives detail”:** Any recovered “sharpness” after video compression usually comes from priors (e.g. edge hallucination) rather than actual information.  We should treat it as artifactual unless proven by analysis.  

6. **Reconstructable degradation regimes:**  - **Mild alias + mild blur:** If LR images contain aliased high freq (due to slightly undersampled sensor) and blur is not too severe (PSF has broad support), multi-frame SR can extend detail.  - **Noise-dominated but static scenes:** Can be denoised via temporal averaging (not truly “super-resolved” but visually cleaner).  - **Moderate motion + misalign:** If misalignment is small and can be compensated, still worth trying SR in local patches.  

7. **Regimes to fallback to spatial-only:**  - **Static frames:** (Sec 8) Purely identical frames.  
   - **Severe blur with no variety:** (e.g. static defocus or a fixed long exposure) – if all frames are equally blurred with zeros, no new info.  
   - **Video with strong temporal compression (e.g. extreme QP, B-frame lag):** In practice, this yields too little reliable alias info. We may decide to ignore multi-frame in extremely low-bitrate regimes.  
   - **Fast scene changes:** Scenes with very little repeat observation (like surveillance with new people every frame) – cannot merge.  

8. **Quantitative gating tests (before costly SR):**  
   - **Shift variance check:** Compute inter-frame motion vectors or alignment flow.  If the residual translation is <0.1 px average, skip SR (no diversity).  
   - **MTF/coherence analysis:** Estimate the empirical MTF of the video stream (e.g. from a known chart).  If it falls off sharply near 0.5× Nyquist, little high-frequency content exists anyway.  
   - **SNR estimate:** If the scene is very dark (low SNR), multi-frame SR will mostly denoise.  Only low-light denoising might be warranted, but extended super-res (hallucination) should be disabled.  
   - **Motion blur detection:** If all frames appear similarly blurred (via blind deconvolution estimation), flag for standard single-frame deblurring rather than multi-frame.  
   - **Codec QP or resolution metadata:** For encoded video, if resolution is already high or QP is high, maybe skip multi-frame upscaling.  

   In essence, we should analytically test for the presence of the required sampling diversity before launching a computationally heavy multi-frame reconstruction.

9. **Open unknowns:**  
   - **Complete model of demosaicing:**  How much high-frequency alias is introduced by Bayer patterns and corrected by demosaic?  This pipeline is complex and content-dependent.  We lack a closed-form for its effect on recoverable luma/chroma detail.  Experiments will be needed.  
   - **Interplay of noise and alias:**  Can a little noise ever help alias resolution (like dithering)?  We suspect it mainly hurts certainty, but the jury is out.   
   - **Adaptive algorithms:**  If we include machine learning in future, we must decide how to separate true recoverable info from hallucination.  That requires a rigorous framework.  
   - **Human factors:**  Ultimately, perceived sharpness may differ from measured bandwidth extension.  Understanding human tolerance for hallucinated vs real detail could influence architectural trade-offs.  

In summary, our architecture should *seek* subpixel motion and varied PSF, *avoid* cases with no sampling diversity, and use clear quantitative checks to govern the strategy.  Only with disciplined gating can we ensure that multi-frame reconstruction exploits *actual recoverable information* and not prior-driven guesses.

## 13. Remaining unknowns  
Despite extensive theory, several critical gaps remain:

- **Combined pipeline analysis:**  Most literature treats one factor at a time.  The *joint* effect of, say, Bayer CFA + JPEG + motion blur + noise on recoverable band is not fully characterized.  We need end-to-end analysis.  
- **Real-world alignment errors:**  Algorithms assume subpixel alignment.  How does residual misregistration (due to optical distortion, rolling shutter, non-rigid motion) quantitatively degrade recoverability?  No simple model exists.  
- **Temporal alias vs motion alias:**  Rapid object motion can create temporal alias (strobing) which is poorly understood in this context.  Can motion interpolation confuse spatial SR?  
- **Variance in natural scenes:**  How often do consumer videos actually meet the aliasing diversity needed?  Some early studies (e.g. scene frequency content) exist, but a large-scale empirical survey is lacking.  
- **Learning vs physics:** If future approaches incorporate learned priors, we must still quantify the difference between learned “detail” and ground truth detail.  Defining a metric for hallucination vs recovery is an open problem.  

These unknowns mean we should be prepared to iterate on design.  The experimental program (Sec. 10) will likely reveal surprises requiring refinement of theory or method.

## 14. Follow-up research  
If time permits after initial testing, we suggest:

- Developing an analytic or machine-checked bound on multi-frame recoverable frequency given a parametric imaging model (e.g. closed-form Cramér–Rao for simple shift+PSF+noise).  
- Creating a benchmark dataset of HR-to-LR video under controlled but realistic conditions for the community (various motion types, blur, codecs).  
- Exploring novel camera designs (e.g. built-in jitter or coded exposure) as proof-of-concept to explicitly inject diversity.  
- Testing perceptual metrics alongside frequency metrics to align technical capability with visual benefit.

## 15. Sources / bibliography  

- Park, S.-C. & Jeon, S.-K. *“Super-resolution image reconstruction: A technical overview,”* IEEE Signal Proc. Mag., May 2003. (models LR image formation).  
- Papoulis, A. (1977). *Generalized Sampling Theorem* (foundational theory).  
- Huang, T.-S. & Tsai, R. (1984). Super-resolution from multiple images (aliasing theory).  
- Wronski, S. et al. (2019). *Handheld Multi-Frame Super-Resolution*, SIGGRAPH 2019 (practical burst-SR limits; natural tremor).  
- Ben-Ezra, M. & Nayar, S. (2005). *“Jitter camera: super-resolution video”*, ICCV 2005 (proof that controlled jitter aids static scenes).  
- Shan, Q. et al. (2008). *“High-quality motion deblurring from a single image”*, SIGGRAPH 2008 (PSF null-filling concept).  
- Google (Pixel team). *Night Sight whitepaper*, Google Research (2019) (smartphone sampling ratio, MF-SR experiments).  
- Shahram, M. & Milanfar, P. (2006). *“SR in noise – limits analysis,”* IEEE Trans. Image Proc. (cited by [74]).  
- Tomko, A. et al. (MERL, 2009). *“Invertible motion blur in video,”* Tech. Report.  
- “Chroma subsampling,” Wikipedia (2026) (overview of 4:2:0, etc.; used for subsampling rationale).  
- Misc. video pipeline references (Optics/MTF): Wikipedia *“Optical transfer function”*, Imatest documentation (online) – used for optical cutoff intuition.  
- MPEG/H.264/HEVC documentation (various) – codec behavior summary from standards, e.g. JBIG.
- Math/Statistic references for information bounds (e.g. Helstrom 1969) as cited in. 

(These notes provide primary evidence and context for the statements above. Citations are to actual papers or authoritative sources where available.)

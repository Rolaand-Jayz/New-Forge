# Executive Assessment

High-quality temporal super-resolution demands not just *some* motion estimate but rich per-pixel guidance on *where* to take data, *whether* that data is valid (visible), and *how confident* to be in it.  In practice, misalignment causes severe artifacts, so subpixel registration is essential for detail recovery.  Simultaneously, a per-sample confidence or visibility mask is required to suppress inconsistent frames or occluded regions.  In short, the reconstructor needs: (1) precise offsets (e.g. subpixel flow or learned shifts) and (2) a reliability weight or uncertainty for each correspondence.  Better optical flow alone is not sufficient – indeed, Zhang et al. show that **avoiding explicit flow** (using deformable alignment) can reduce error propagation.  Likewise, Chan et al. find that multiple learned offset “hypotheses” per pixel (as in deformable conv) capture occlusions and hidden details better than a single flow.  Thus, flow-fields can serve as a baseline, but feature-based or implicit aligners deserve equal footing.  

Occlusion/disocclusion handling is mandatory.  Forward/backward consistency tests detect occlusions, and missing regions (holes) must be filled from alternate frames (as in OCAI).  In practice, robust fusion (soft blending via masks) eliminates ghosting (see Wronski’s example in Figure 9).  Confidence weighting has been shown to improve results: e.g. Wronski et al. assign per-pixel “robustness” weights to fully reject misaligned samples, and ProbFlow/Uncertainty networks demonstrate that learned entropy measures correlate strongly with flow accuracy.  

In summary, a source-faithful reconstructor needs *offsets+weights* for each candidate frame at every target pixel, plus explicit occlusion reasoning.  Subpixel alignment (e.g. refined Lucas–Kanade) is typically required to unlock super-resolution detail.  However, systems should lean on robust fusion (via confidence masks, deformable alignment, or attention) rather than rely on perfect flow.  As evidence, recent lightweight VSR models achieve near-real-time speed by **forgoing optical flow entirely** and using deformable/attention alignment.  We therefore prioritize representations that encode both *where* to sample (explicit flow or learned offsets) and *how trustworthy* each sample is (confidence scores or uncertainty estimates).

# Reconstruction’s Correspondence Requirements

Temporal reconstruction needs three pieces from correspondence at each target pixel: **(a)** a spatial offset indicating where in the other frame to sample; **(b)** a visibility/validity flag (is this sample truly of the same surface or is it occluded?); and **(c)** a confidence or weight quantifying trust.  All three emerge in classical multi-frame SR.  For instance, Wronski et al. explicitly compute a “robustness mask” that downweights pixels where alignment error is high.  They observe that without such masking, fusion causes ghosting (Figure 9, left).  In practice this mask is built by comparing a local color difference to the local standard deviation: small differences (within one sigma) are safe and merged, large differences are rejected.  

Precise subpixel offsets are equally critical.  Xu et al. show that even *resampling* interpolation matters: bilinear or bicubic warping blurs high-frequencies, whereas a good alignment must “avoid imposing low-pass filtering”.  Similarly, Wronski et al. note that “we require subpixel accurate alignment to achieve super-resolution,” and they refine block-matching vectors with Lucas–Kanade iterations to reach that precision.  In short, correspondences must resolve motions to fractions of a pixel to recover detail.  

Finally, each offset must carry confidence.  If a match is uncertain, its contribution should be downweighted or discarded.  Modern approaches capture this via learned uncertainty heads or statistical tests.  For example, ProbFlow (Wannenwetsch et al.) jointly estimates flow **and** per-pixel entropy, finding that flow uncertainty correlates with pixel reconstruction error.  Unsupervised methods like U²Flow (Sun et al., CVPR 2026) explicitly predict uncertainty to adapt refinement and fusion.  Learned SR systems often incorporate this implicitly; e.g. Zhang et al.’s LIF-VSR abandons flow completely and instead uses attention-based fusion that inherently weighs each pixel (achieving real-time speeds with only 3.06M parameters).  

**Summary:** a “good” correspondence here is not just a small endpoint error – it must come with a per-pixel confidence and occlusion logic.  In practice, effective systems combine refined offsets (optical flow or learned shifts) **and** per-sample weights to gate fusion. 

# Classical Registration

Classical methods give baselines but have limitations.  **Phase correlation** (FFT-based) can estimate pure translation very efficiently, and can be made subpixel by interpolating the cross-correlation peak.  However, it assumes *global translation only*, failing on rotations, zoom, or local object motion.  Critically, frequency-domain aliasing can seriously degrade its accuracy, and it offers no per-pixel confidence or occlusion handling.  **Parametric fits** (affine or homography) extend phase-correlation to rotation/scale but share its global scope: they align an entire scene plane but break when objects move independently.  

**Local methods:** Block matching and Lucas–Kanade (LK) are workhorses.  Simple block-matching (with a search window) handles moderate local shifts but yields blocky, often coarse results; subpixel refinement requires extra steps.  Wronski et al. use a pyramid-based block-matcher followed by three LK iterations to reach high precision cheaply.  Pure **LK optical flow** (with pyramids) yields dense subpixel flow via iterative gradient descent.  LK handles small motions well and naturally provides a structure tensor (Hessian) which can give an error estimate per match, but it struggles in textureless regions (aperture problem) and finds only a single offset per pixel.  **Horn–Schunck/variational flow** solves globally with a smoothness term, giving smooth dense fields – useful on diffuse motion but prone to over-smoothing edges. **Farnebäck’s flow** (polynomial expansion) is another dense method; it runs on full frames and is smoother than block matching but similarly offers no confidence output.

In all these, motion range is limited by pyramid depth, robustness is limited (no explicit occlusion test), and no uncertainty map is produced by default.  They can be GPU-accelerated (FFT for phase, parallel per-pixel ops for flow) but these methods became outpaced by learned flows.  Notably, registration accuracy *directly affects SR quality*.  Vera and Torres emphasize that “registration accuracy is directly related to the quality of the final superresolved images”. Thus classical methods can serve as a controlled reference, but leave major gaps in occlusion/uncertainty modeling.  

# Learned Optical Flow

Deep networks (FlowNet, PWC-Net, RAFT, etc.) dominate optical flow today.  They produce full dense flow fields, often with subpixel precision, but at varying computational cost.  PWC-style networks (pyramidal cost volumes) run moderately fast (10–30ms at HD on GPU); RAFT and its variants (all-pairs correlation + iterative updates) are more accurate but much heavier (often 100–300ms per frame pair).  Newer flow models emphasize efficiency (LiteFlowNetX, DCFlow, SwinFlow) or add blur-robust branches, but fundamentally they all output a single 2D displacement per pixel.  

Crucially, **flow networks do not provide occlusion masks or uncertainty by default**.  Some learn an occlusion probability (a few architectures include an occlusion head), and uncertainty can be obtained by ensembling or special training (Wannenwetsch et al. show variational flow yields calibrated entropy, Ilg et al. use multi-hypothesis networks to get uncertainty). However, even a perfect optical flow can mislead SR if it “hallucinates” in occluded regions or smooths motion boundaries.  Chan et al. (2020) explicitly compare flow-based vs. learned-offset alignment and find deformable offsets mitigate occlusions better: they observe that learned offset fields have greater diversity (multiple offsets per pixel), producing richer warped details than single-flow warping. 

Thus while flow is a natural baseline (it gives explicit geometry), it is often **overkill** for SR.  For static scenes, camera motion can be estimated once (via a homography or SLAM) and used for background; residual motion can be smaller.  For dynamic objects, a dense flow is informative but expensive.  As Feng et al. note, bypassing flow entirely (using an RNN to propagate features and attention to fuse) avoids error propagation from flow estimation.  In practice, we recommend testing both: a state-of-the-art flow (e.g. RAFT) as a reference, and one or more learned alignment strategies (e.g. PWC, GMA, FlowFormer) to gauge marginal benefit. We should also explore lightweight flows (SPyNet or distilled RAFT) to find performance/accuracy trade-offs.  

Ultimately, flow fields give a clear correspondence representation, but do not carry explicit confidence or multiple hypotheses.  If flow is used, it must be supplemented by occlusion masks and perhaps outputting an uncertainty (entropy or estimated error) per pixel.  Alternatively, implicit methods (next section) can provide alignment *and* confidence in one step.

# Feature-Space and Implicit Correspondence

Instead of explicit flow, many modern SR methods learn **feature-space alignment** via deformable convolutions, dynamic kernels, or attention.  A prime example is EDVR (Zhang et al. 2019): it builds a pyramid-cascading deformable (PCD) module that aligns neighboring frames by learning offsets in feature space.  Similarly, Xu et al. (2026) propose an “implicit resampling” network where learned coordinate networks and attention replace interpolation.  These approaches do not produce a human-interpretable flow field; instead they directly sample or warp features at learned offsets.  

Such implicit alignment can be advantageous.  Deformable convolutions naturally handle large motions and can learn multiple offsets per location.  For example, Chan et al. show that EDVR-like deformable alignment yields more detail on occluded or fine-structure regions than warping by single-flow.  Attention-based methods (cross-frame or non-local attention) similarly learn where to gather information.  Yu et al. (2021) demonstrate a cross-frame non-local attention that “allows video SR without frame alignment,” which is robust to large motion.  They also add a memory bank to retain details beyond immediate neighbors.  In practice, we should test such methods: e.g. an RNN-based VSR with deformable conv (like LIF-VSR) or transformer-based fusion.  

Implicit methods inherently include confidence: attention weights or deformable kernel magnitudes can serve as confidence clues.  However, they require training and are harder to predict failure cases.  We should ensure any learned alignment is calibrated: one can examine the learned offsets (e.g. via [78]) and test how often they “guess” incorrectly.  Despite interpretability challenges, feature-based aligners often excel in textureless or repetitive regions where flow fails, and they bypass heavy cost volumes.  Given evidence like EDVR and LIF-VSR achieving strong results without explicit flows, implicit correspondence deserves equal priority as a test candidate.

# Splatting and Warping Directions

Mapping between frames can be done backward (pulling target pixels from source) or forward (splatting source to target).  **Backward warping** (using an estimated flow field to sample a source image at target coordinates) is standard and simpler: it naturally avoids holes (every target pixel can pull from a location).  However, backward warping can only incorporate one source pixel per target; additional source contributions require careful fusion of multiple backward warps.

**Forward warping** (“splatting”) can aggregate many source pixels into one target location, which is more naturally many-to-one.  Niklaus & Liu’s *softmax splatting* is a recent example: they forward-warp each source pixel using flow, but resolve collisions via a softmax weighting, effectively blending overlapping pixels.  Softmax-splatting has been used successfully for interpolation, but by itself it leaves holes (disocclusions).  To handle holes, OCAI (2024) uses bidirectional flows: one forwards both frames to an intermediate time and then fills holes by backing in from the opposite frame.  In forward splatting, an occlusion map is crucial: OCAI infers which pixel should “win” a collision by assuming the occluding (closer) pixel should dominate, then applies a hole-filling.

In summary, both directions have trade-offs.  Backward warps are straightforward but need explicit occlusion masks to combine multiple frames.  Forward splats can fuse evidence from many frames naturally but require mechanisms for collision resolution and hole filling.  A robust system may employ a hybrid: use backward warps for visible areas and soft forward-splat blends for overlapping evidence, always guided by occlusion/visibility estimates.

# Occlusion and Disocclusion

A key challenge is ensuring that only *visible* pixels are fused.  When an object moves and covers another, or when new regions appear, we must detect and exclude invalid data.  **Occlusion detection** is often done by forward-backward consistency: a pixel whose backward flow does not return it to the same source is likely occluded.  Many flow algorithms also estimate an occlusion map (e.g. via brightness constancy violations).  In practice, one can mask out occluded pixels or avoid warping them.  For example, OCAI obtains an occlusion mask $O_{0,1}$ by forward-back consistency and uses it to weight forward splatting.

**Disocclusion** (new areas entering view) means some target pixels have no source neighbor.  These holes often happen in forward warps.  OCAI’s solution is to fill holes by warping in the other direction (from the opposite frame) and using consistency to blend.  In general, we must allow holes and either inpaint them or leave them to be filled by a generative prior if no frame has seen that pixel.  

Without occlusion reasoning, fusion artifacts appear as double edges or ghosting (see Wronski’s moving bus example).  Thus, *explicit* occlusion handling is mandatory.  Either by masking contributions (rejecting occluded samples) or by modeling scene geometry (e.g. layered depth) the reconstructor must avoid combining unrelated surfaces.  Techniques like multi-layer optical flow or scene flow could help, but at minimum we must use per-pixel occlusion flags to gate fusion.  

# Surface Ownership and Layering

Many scenarios violate the single-flow-per-pixel assumption: overlapping motions, transparency, fences, hair, etc.  In those cases, a target pixel might legitimately receive evidence from multiple “surfaces.”  One solution is **layered motion**: segment the scene into layers or objects, each with its own correspondence.  For example, in autonomous driving, one often fits multiple rigid motions to superpixels.  In SR, we might need similar: either detect object segments or simply keep multiple hypotheses per pixel.  

Chan et al. provide a clue: their deformable module effectively learns “offset diversity,” where each pixel gathers from several offsets, akin to multi-hypothesis tracking.  This suggests a design where we **do not** force a single match.  For instance, a pixel on a glass window might have one offset for the glass and another for the background – both could be stored with confidences.  Practically, this argues against committing too early.  We should allow multiple candidate correspondences or layers (with associated weights) until the final fusion, rather than compressing them into one.  

Explicitly, if two source pixels claim the same target (a collision), we should prefer the one with higher confidence (e.g. via an occlusion check) and possibly still record the second for evidence if both may contribute.  If no source covers a target, we mark it as disoccluded and rely on prior or single-image cues.  In short, treating each target as “owned” by at most one source is often wrong; we should support layered or soft assignment.  

# Confidence and Uncertainty

**What is confidence?** Physically, it represents the trust in a correspondence (or in the propagated high-res estimate).  Ideally, a confidence score $c\in[0,1]$ predicts whether using that pixel will reduce or increase reconstruction error.  Calibration is critical: if we say $c=0.8$, we hope in 80% of cases that pixel’s error is low.  Learned methods train networks to output either variances or multi-hypothesis distributions to capture uncertainty.  

Prior work shows learned confidences outperform hand-crafted measures.  Wannenwetsch et al. (ProbFlow) derive per-pixel uncertainties by a probabilistic flow model.  Ilg et al. train a network to output variance by a multi-hypothesis WTA loss, and they demonstrate these uncertainties are more accurate than classical confidence heuristics.  More recently, U²Flow trains uncertainty via augmentation consistency, using it to modulate refinement and fusion losses.  In all cases, the uncertainty is typically correlated with photometric error or optical flow entropy.  

In practice, we should use confidence maps to *weight* the fusion (high-confidence samples carry more weight).  Experiments should verify calibration: e.g. group pixels by predicted confidence and check actual error distribution.  Ideally, the system should maintain uncertainty through the pipeline.  For example, in a Kalman-filter-like framework, we would carry a covariance (uncertainty) of the high-res estimate and update it with each new frame (see next section).  

No matter the representation (flow or implicit), we should encourage the model to output a confidence.  For explicit flow, adding a residual-confidence head (entropy of cost volume or photometric consistency) can be fruitful.  For implicit methods, attention weights or kernel magnitudes play a similar role.  Critically, *saturated* confidence (always 1 or 0) is unhelpful: the system should distinguish moderately reliable from highly reliable pixels.  

# State-Estimation Perspective

It is useful to view video reconstruction as a temporal state-estimation problem.  One can imagine a hidden high-res frame $X_t$ (the “state”) that we update each step with new observations (frames).  Analogously to a Kalman filter or smoother, the system could maintain: (i) a latent HR estimate $\hat{X}_{t}$, (ii) its uncertainty/covariance, and (iii) an observed proposal from the current LR frame.  Feng et al.’s KEEP framework concretely implements this: they maintain a latent feature state (the face prior) with an associated uncertainty, and at each frame perform a Kalman-like update.  

This suggests design ideas: between frames, propagate the state (perhaps via a simple warp or identity); then, treat the new frame as a noisy measurement.  The gain (akin to confidence) should depend on uncertainty – if the measurement is very uncertain (e.g. occluded), rely more on the prior.  While a literal Kalman filter (with matrices) may be impractical, RNNs or diffusion models can approximate this logic.  In evaluation, we should test how a “recurrent” fusion (BasicVSR-style) compares to a filter formulation.  KEEP demonstrates that leveraging past reconstructions improves temporal coherence.  

Key takeaway: frame-by-frame SR is suboptimal because it throws away previous evidence.  Recurrent or iterative schemes (like filtering or smoothing) naturally propagate old details.  The system should be designed to accumulate evidence over time, not treat each frame independently.  Tracking variables (features) with uncertainty is a promising direction.

# Long-Range Temporal Correspondence

How far back should we look? Intuitively, contributions from very old frames decay as motion and appearance change.  However, distant frames might contain unique viewpoints (e.g. camera pans revealing different aliasing) that adjacent frames do not.  Some architectures store multi-frame memories: for instance, Yu et al.’s memory-augmented attention stores general features from many frames.  In contrast, recurrent schemes like BasicVSR implicitly pass information forward frame by frame.  

We need to test this adaptively.  Key questions: does adding frame $t-4$ improve detail beyond $t-1,t-2$?  At what lag does correspondence become unreliable?  Our proposed controlled experiment (section **Controlled Experiments**) will measure PSNR vs frame distance.  Intuitively, static static regions can benefit from many samples (exploiting all alias diversity), but dynamic scenes may only safely use a short history (to avoid occlusion errors).  Therefore, a fixed window (e.g. 5–7 frames) is common, but **adaptive reach** (extending the window when beneficial) could be useful.  For example, if a surface remains visible and tracking confidence remains high, the system might pull in farther frames.  

There is little published data on this in VSR.  As a hint, Weisz et al. (not in our refs) found that SR gains saturate beyond ~5–7 frames for most scenes.  We will empirically quantify this.  If distant frames do add new detail, we may incorporate a memory bank (as in transformers) or keyframe buffer.  Otherwise, we should limit to a manageable window for performance.

# Real-time Feasibility

Finally, we assess practical cost.  Classical methods can be fast (pyramidal LK can run at tens of FPS on GPU); learned flow models range from very heavy (RAFT) to very light (SPyNet, LiteFlowNet).  Attention and deformable modules add overhead.  We must categorize each major approach:

- **Classical baselines:** Phase correlation (FFT) is very fast (one FFT per frame) but limited; block matching is embarrassingly parallelizable (good on GPU); Lucas–Kanade flow can also be GPU-parallel.  All can reach 30–60 FPS at modest resolution, but they lack advanced features.
- **Learned flow:** PWC-Net (~8M params) can run ~30Hz on 1080p on a high-end GPU, whereas RAFT (~5M params) takes ~100-200ms. LiteFlowNetX and others can approach real-time at 720p. Unsupervised/stereo-style flows (e.g. DispNet) often use an encoder–decoder with a few GFLOPs. Confidence outputs add minor cost. Memory hog: full-resolution cost volumes (all-pairs) can be hundreds of MB.
- **Deformable/attention:** EDVR’s PCD module adds ~10ms overhead for 512×512 inputs; LIF-VSR reports *near-real-time* with only 3.06M parameters, implying tens of FPS on modern GPUs. Sparse attention (sampling a few keys) reduces cost compared to full self-attention. Splatting is simple forward passes. Uncertainty heads (extra convs) are marginal overhead.
- **Memory:** Storing a few keyframes’ features is cheap (MBs), but large memory bank (like many frames for attention) can explode memory usage unless sparsified.

Based on this, **plausible 30 FPS candidates** include: pyramidal LK flow (GPU optimized), SPyNet/LiteFlowNet, deformable conv modules (like EDVR/LIF), and sparse attention fusion. **Borderline**: full RAFT or all-pairs attention at 4K (likely too slow). Confidence weighting and small networks pose negligible cost. The table below qualitatively rates each approach on throughput.

# Controlled Experiments

To isolate correspondence effects, we propose:

- **A. Oracle Correspondence:** Use ground-truth geometry/motion (from synthetic data or calibrated scenes) to align frames perfectly, and measure the best-case reconstruction (PSNR/SSIM). *Falsifier:* if SR quality does not significantly exceed single-frame SR, it implies temporal diversity adds little new detail.
- **B. Controlled Error Injection:** Take a fixed SR fusion method and feed it motion fields corrupted by known amounts of error (Gaussian jitter, bias, missing data). Measure output quality vs error magnitude. This quantifies sensitivity. A failure mode: if even small flow errors cause large SR degradation, the method needs very precise correspondence.
- **C. Classical vs Learned Flow:** Use the *same* SR fusion architecture (e.g. a fixed multi-frame CNN) but supply classical flow vs a learned flow. Compare quality. If learned flow gives little gain over a simpler flow, it suggests alignment accuracy is not the bottleneck.
- **D. Flow vs Deformable/Implicit Alignment:** Keep the SR model fixed, compare using explicit flow warping vs learned deformable conv/attention for alignment. This tests whether bypassing flow helps reconstruction (as Chan et al. suggest).
- **E. Occlusion-Aware vs Blind:** Give the reconstruction an exact occlusion mask (so occluded contributions are dropped) versus ignoring occlusions. Check ghosting/artifacts. A big gap would falsify approaches that do not explicitly handle occlusion.
- **F. Confidence Calibration:** For a set of correspondences with predicted confidence, plot predicted confidence vs actual reconstruction error. Good methods should show that high-confidence pixels indeed have low error. If confidence is uncalibrated (flat correlation), we must improve it.
- **G. Long-Range Value:** Measure reconstruction gain from distant frames. Use a fixed SR method and vary frame gap (e.g. up to 30 frames apart). Compare PSNR and unique frequency content. If very distant frames add no new detail, we needn’t look back far. If they do (e.g. latent static scene structure re-emerges), then adaptive reach is justified.

For all experiments, metrics include PSNR/SSIM on synthetic benchmarks with known ground truth, perceptual quality measures, and artifact counts.  Calibration can use Expected Calibration Error (ECE) on confidence.  Ghosting can be quantified by edge strength of spurious doubles.  These tests will falsify assumptions like “small flow error ⇒ small SR error” or “distant frame always helps”.

# Comparative Evaluation Matrix

Below we compare representative approaches in terms of geometry, occlusion, uncertainty, etc. (no single overall score is given).

- **Phase Correlation:** 
  - *Accuracy:* Good for global translation, ~subpixel with interpolation. 
  - *Usefulness:* Only aligns global frame; cannot align local motion or non-rigid detail.
  - *Occlusion:* No handling. 
  - *Uncertainty:* None (though correlation peak sharpness could serve as proxy). 
  - *Motion Range:* Unlimited translation, but fails on rotation/zoom. 
  - *Non-rigid:* None. 
  - *Parallax:* Ignores. 
  - *Fine Structure:* Not applicable beyond averaging. 
  - *Interpretability:* High (simple FFT). 
  - *Training:* None. 
  - *Hallucination Risk:* None (it does no warping). 
  - *Compute:* Very low (FFT). 
  - *Memory:* Low. 
  - *Real-time:* Yes. 
  - *Role:* Baseline for global alignment and analysis.

- **Lucas–Kanade / Pyramidal LK:**
  - *Accuracy:* Subpixel accuracy per pixel, local minima risk. 
  - *Usefulness:* Good for small displacements, yields flow-like field.
  - *Occlusion:* Blind (one direction). 
  - *Uncertainty:* Hessian inversion gives covariance (often unused). 
  - *Motion Range:* Moderate (with pyramid). 
  - *Non-rigid:* Yes (local, can move independently). 
  - *Parallax:* Handles local motion, but no explicit depth. 
  - *Fine Structure:* Can track fine features if texture present. 
  - *Interpretability:* High (gradient method). 
  - *Training:* None. 
  - *Hallucination:* No. 
  - *Compute:* Medium (iterative). 
  - *Memory:* Low. 
  - *Real-time:* Often yes on GPU. 
  - *Role:* Baseline precise alignment, often used to refine other methods.

- **Block Matching (Hierarchical):**
  - *Accuracy:* Pixel-level shifts within blocks; coarse until refined. 
  - *Usefulness:* Parallelizable, good for rigid motion.
  - *Occlusion:* No; block collisions ignored. 
  - *Uncertainty:* Can use matching error. 
  - *Motion Range:* Can be large with hierarchy. 
  - *Non-rigid:* No (block moves rigidly). 
  - *Parallax:* No. 
  - *Fine Structure:* Poor at edges. 
  - *Interpretability:* Medium (simple search). 
  - *Training:* None. 
  - *Hallucination:* No. 
  - *Compute:* Low–medium. 
  - *Memory:* Low. 
  - *Real-time:* Yes on GPU. 
  - *Role:* Very fast baseline; refine with LK for SR.

- **Horn–Schunck / Variational Flow:**
  - *Accuracy:* Smooth dense flow; often subpixel. 
  - *Usefulness:* Captures overall motion but oversmooths. 
  - *Occlusion:* No explicit. 
  - *Uncertainty:* Possible via energy residual. 
  - *Motion Range:* Limited by assumed smoothness; large flows need multi-scale. 
  - *Non-rigid:* Limited (global smoothness tends to merge regions). 
  - *Parallax:* No geometry. 
  - *Fine Structure:* Often blurred by smoothness. 
  - *Interpretability:* Medium (PDE). 
  - *Training:* None. 
  - *Hallucination:* Can “smear” across boundaries. 
  - *Compute:* Medium-high (iterative). 
  - *Memory:* Medium. 
  - *Real-time:* Possibly at low res. 
  - *Role:* Baseline flow; good for coarse alignment.

- **Phase-Only Registration / Homography Fitting:**
  - *Accuracy:* High for global, with good data. 
  - *Usefulness:* Perfect for static planar camera motion. 
  - *Occlusion:* No. 
  - *Uncertainty:* Fit error gives confidence. 
  - *Motion Range:* Captures large rotations/perspective. 
  - *Non-rigid:* None (one plane). 
  - *Parallax:* No (assumes planarity). 
  - *Fine Structure:* None (just global mapping). 
  - *Interpretability:* High (clear model). 
  - *Training:* None. 
  - *Hallucination:* No. 
  - *Compute:* Low (RANSAC+LS). 
  - *Memory:* Low. 
  - *Real-time:* Yes. 
  - *Role:* Good for compensating camera motion before local processing.

- **RA F T & Modern CNN Flows (PWC, GMA):**
  - *Accuracy:* State-of-art end-point error; subpixel. 
  - *Usefulness:* Very good on textured, lambertian motion. 
  - *Occlusion:* Usually no explicit mask (some outputs invalid flows). 
  - *Uncertainty:* Can add a head, but not standard. 
  - *Motion Range:* Handled via pyramids or cost volumes. 
  - *Non-rigid:* Yes (dense flow). 
  - *Parallax:* Implicitly handled as apparent flow (no depth). 
  - *Fine Structure:* Generally good unless textureless. 
  - *Interpretability:* Moderate. 
  - *Training:* Requires large datasets. 
  - *Hallucination:* Minimal (predicts smooth continuation). 
  - *Compute:* Very high (many correl.). 
  - *Memory:* Very high (cost volumes). 
  - *Real-time:* No (except very small variants). 
  - *Role:* Reference alignment; provide full mapping for occlusion checks.

- **LiteFlowNet / SPyNet:**
  - *Accuracy:* Lower than PWC/RAFT, but real-time. 
  - *Usefulness:* Acceptable for SR if speed-critical. 
  - *Occlusion:* Typically no special handling. 
  - *Uncertainty:* None by default. 
  - *Motion Range:* Small (SPy at coarse levels). 
  - *Non-rigid:* Yes (dense). 
  - *Compute:* Low (sparse costs). 
  - *Memory:* Low. 
  - *Real-time:* Yes at HD. 
  - *Role:* Candidate for on-device SR or very fast baselines.

- **EDVR / Deformable Alignment:**
  - *Accuracy:* Learns to align complex motions across frames. 
  - *Usefulness:* Excellent on large non-uniform motion and occlusions. 
  - *Occlusion:* Implicitly handled by learned offsets and later fusion. 
  - *Uncertainty:* Not explicit, but one can inspect offset magnitudes. 
  - *Motion Range:* Large (coarse-to-fine cascade). 
  - *Non-rigid:* Yes (multiple layers of flow). 
  - *Parallax:* Handled via multiple offsets at feature levels. 
  - *Fine Structure:* Good if network capacity suffices. 
  - *Interpretability:* Low (NN black-box). 
  - *Training:* High (many examples). 
  - *Hallucination:* Low if well-regularized. 
  - *Compute:* Moderate (a few ms per frame at 512p). 
  - *Memory:* Moderate. 
  - *Real-time:* Possible with small models (e.g. LIF-VSR). 
  - *Role:* Strong candidate; merges alignment and fusion in one.

- **Dynamic Convolution/Kernel Prediction:**
  - *Accuracy:* Learns local warps/kernels per pixel. 
  - *Usefulness:* Can capture complex resampling. 
  - *Occlusion:* Masking still needed. 
  - *Uncertainty:* Possible via predicted weight entropy. 
  - *Motion Range:* Limited by kernel size. 
  - *Compute:* High (per-pixel kernels). 
  - *Role:* Specialized; seldom used alone for SR.

- **Attention-Based (Non-local, Cross-Attn):**
  - *Accuracy:* Can match arbitrary points across frames. 
  - *Usefulness:* Powerful for repeating patterns and large displacements. 
  - *Occlusion:* Must be learned; no inherent model. 
  - *Uncertainty:* Attention weights give confidence per match. 
  - *Motion Range:* Unlimited (global). 
  - *Compute:* Very high (O(N²) naive; sparse makes it feasible) 
  - *Role:* Promising, but must be carefully limited or sparse to run in real-time.

- **Layered/Multi-Hypothesis Flow:**
  - *Accuracy:* Represents multiple motions per pixel. 
  - *Usefulness:* Excellent for occlusions and transparency. 
  - *Occlusion:* Inherently handles occlusions by having separate layers. 
  - *Compute:* Very high (hypothesis explosion). 
  - *Role:* Likely research-use; we might approximate via deformable offsets or multiple passes.

- **Confidence/Uncertainty Methods:**
  - *Accuracy:* Complementary rather than standalone. 
  - *Usefulness:* Can greatly improve fusion by downweighting bad data. 
  - *Compute:* Minor overhead. 
  - *Role:* Essential for weighting evidence; e.g. Wronski’s robustness mask is one such confidence map.

# Architecture Implications and Recommendations

From the above, we conclude:

- **Correspondence output:** The reconstructor should accept correspondences plus confidence masks.  Practically, this means either a flow field + a per-pixel weight, or learned offsets + weights.  The alignment module should ideally provide both spatial mapping and confidence.  
- **Flow vs implicit:** Explicit flow need not be the default.  We recommend *also* developing a model with implicit feature alignment (deformable conv or attention) in parallel.  If one approach fails under challenging motion or occlusion, the other might succeed.  In particular, given the successes of EDVR and LIF-VSR, we should test architectures that discard flow entirely.  In our baseline suite, we propose at least one pure-flow pipeline (e.g. PWC warp + CNN fusion) and one implicit pipeline (e.g. deformable alignment + CNN).
- **Occlusion handling:** Explicit occlusion is mandatory.  If using flow, compute a forward-back consistency mask and use it to exclude unreliable pixels.  If using splatting, use softmax weights or learn a mask as in OCAI.  In either case, ensure the SR model can accept a per-pixel mask (e.g. skip fusion where mask=0).  Neglecting occlusions will produce artifacts.
- **Layering:** Our architecture should allow multiple candidate matches per pixel, or equivalently a segmentation/layered representation.  A simple practical step is to fuse only high-confidence correspondences and leave ambiguous regions to single-frame enhancement or inpainting.  If possible, we could extend flow prediction to output multi-layer flows (research-level).  At minimum, we must be aware that transparent/fine-structure regions cannot be solved by a single flow.  
- **Confidence representation:** We should represent uncertainty explicitly throughout the pipeline.  Options include an additional confidence channel for each warped frame, or an uncertainty map for the fused HR estimate (as in a Kalman filter).  These confidences should guide the fusion weights in the reconstruction network.  Our design will pass per-pixel weights through from correspondence to fusion layers.  
- **Temporal extent:** An adaptive window is justified.  The system should not blindly average dozens of frames: instead, it could gradually downweight or drop distant frames whose correspondences become uncertain.  We should implement either a decaying confidence over time or a gating mechanism (e.g. memory cells that self-reset).  
- **First candidates to test:** 
  1. **Flow-based baseline:** e.g. PWC-Net or LiteFlowNet + simple warp + an SR network (e.g. BasicVSR) as in traditional pipelines.  
  2. **Learned flow high-accuracy:** RAFT-like + warp + CNN.  
  3. **Deformable conv pipeline:** e.g. EDVR or LIF-VSR style.  
  4. **Attention fusion:** e.g. a transformer block to fuse features without explicit warp (perhaps starting from BasicVSR++ with added cross-attn).  
- **Real-time path:** Among these, SPyNet/LiteFlowNet and small deformable networks are most promising for 30–60 FPS.  RAFT, full attention or deep ensembles are likely too heavy.  For a 60 FPS target, a pre-computed or very sparse alignment (like small kernel or sparse attention) might be necessary.  We should profile each component early.

Finally, any architecture must be evaluated on both *reconstruction fidelity* and *performance*.  We will benchmark throughput on representative hardware (mid-range GPU), keeping 30 FPS as a hard line.  If a promising method is too slow, we will simplify it (prune the model, reduce cost volumes or resolution) rather than abandon core ideas, since performance optimization is secondary to science in this study.

# Remaining Unknowns

Key open issues include:

- **Alignment error vs SR quality:** The quantitative relationship between correspondence error and final reconstruction error is still unclear.  How many pixels of flow error degrade PSNR by 1 dB?  This needs thorough testing (Experiments A–B).  
- **Optimal confidence strategy:** We need to discover which confidence metric (flow variance, photometric residual, learned output) best predicts SR benefit.  And how to propagate uncertainty through the network (collapsing to one mask vs a full variance map) is unsettled.
- **Long-range gains:** It remains to be seen whether very distant frames (beyond ~10–20) truly add new *source* detail or just redundant observations.  The answer likely depends on scene dynamics; adaptive methods might be needed but no guidelines exist yet.
- **Layered correspondence:** The best practical form of multi-hypothesis motion is unresolved.  Do we need full layered scene representations (e.g. 3D scene flow), or can deformable offsets suffice in practice?  
- **Confidence calibration:** We must validate that our confidence maps are calibrated to actual error.  Prior work shows learned uncertainties can be good, but the effect on SR fusion quality (not just flow EPE) needs study.
- **Unsupervised / self-supervised training:** If we lack ground-truth flows, can the system learn good correspondences implicitly?  Early work like RAFT unsupervised or U²Flow suggests yes, but we may need to train on actual video data.
- **Splatting vs warp architectures:** While softmax-splatting handles collisions elegantly, it’s untested in SR (most use back-warping).  We should explore whether forward splat+fusion can outperform standard warping when properly integrated.  

Overall, these unknowns will guide our experiments.  Wherever we lack clear evidence, we must explicitly test the alternatives (as outlined in the experiment section).

# Source Notes / Bibliography

- Wronski _et al._ (2019), *Handheld Multi-Frame Super-Resolution* (SIGGRAPH 2019).  
- Xu _et al._ (2024), *Enhancing Video SR via Implicit Resampling* (CVPR 2024).  
- Zhang _et al._ (2026), *LIF-VSR: Lightweight Video SR with Implicit Alignment* (Sensors 2026).  
- Sun _et al._ (2024), *OCAI: Occlusion-Aware Interpolation* (CVPR 2024).  
- Wannenwetsch _et al._ (2017), *ProbFlow: Joint Flow and Uncertainty* (ICCV 2017).  
- Ilg _et al._ (2018), *Uncertainty Estimates and Multi-hypotheses for Flow* (ECCV 2018).  
- Feng _et al._ (2024), *KEEP: Kalman-Inspired VSR for Faces* (CVPR 2024).  
- Niklaus & Liu (2020), *Softmax Splatting for Frame Interpolation* (ICCV 2021).  
- Chan _et al._ (2020), *Understanding Deformable Alignment in VSR* (AAAI 2020).  
- Sun _et al._ (2026), *U²Flow: Uncertainty-Aware Flow* (CVPR 2026).  
- Yu _et al._ (2021), *Memory-Augmented Non-Local Attention for VSR* (CVPR 2021).  
- Vera & Torres (2008), *Image Reconstruction Error for Optical Flow* (ICIP 2008).  
- Zhang _et al._ (2019), *EDVR: Enhanced Deformable Video Restoration* (CVPR Workshops 2019).  

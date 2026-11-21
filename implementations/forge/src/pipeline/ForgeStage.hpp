// ForgeStage — FORGE (docs/design/FORGE-NEW-MODEL.md), the first-party
// lanczos3-conditioned correction net, as an opt-in pipeline stage "forge":
//
//   out = clamp(lanczos3(in) + delta(in), 0, 1)     linear light, 3x, rgba16f
//
// delta = ForgeDeltaNet (tools/forge_model.py) on x = srgb_encode(in) - 0.5:
// stem 3x3 -> 12 fused dw3x3+pw1x1 blocks -> 1x1 tail + pixel shuffle. GPU
// path = shaders/forge_stem.comp, forge_block.comp (x12), forge_tail.comp
// (tail + shuffle + lanczos3 base + composite). Feature maps live in two
// ping-pong storage buffers (fp16 storage by default, fp32 selectable via the
// "storage" param or ANVIL_FORGE_STORAGE=fp32); all arithmetic is fp32.
//
// Default stage: main.cpp adds it when no ANVIL_PIPELINE and no variant is
// set (ANVIL_FORGE_DEFAULT=0 opts out). Inputs above ANVIL_FORGE_MAX_PIXELS
// (default 1280x800 = 1024000) pass through unchanged: 3x of a larger frame
// is beyond 4K and 1080p costs ~10 ms (measured 2026-11-15).
//
// Weights: .forge v1 file (tools/forge_export.py documents the format) from
// env ANVIL_FORGE_WEIGHTS, else ~/.local/share/anvil/forge/default.forge,
// else <repo>/assets/forge/merged-v2.forge (the hand-constructed merged
// model, tools/forge_construct.py --mode merged). Tell-line at load:
//   "forge: weights loaded <path> sha256=<hex16 of the file> params=<n>"
// Missing/corrupt weights -> "forge stage disabled: <reason>" and the stage
// passes its input through unchanged (1:1).
//
// Tuning (experiment knobs; defaults are the fastest measured):
//   ANVIL_FORGE_FUSE=1|2|3 (default 1): body blocks fused per dispatch (fp16
//     storage with LDS staging only; else 1 — shared-memory budget).
//   ANVIL_FORGE_PPT=1|2|4 (default 1): block-kernel pixels per invocation.
//   ANVIL_FORGE_STAGE=0|1 (default 1): stage tile+halo in shared memory.
// ANVIL_FORGE_TIMING=1: GPU timestamps (own query pool, per frame in flight)
// around the whole dispatch sequence + per kernel; logs median/p90 every
// ANVIL_FORGE_TIMING_N (default 100) frames.
//
// Temporal path (default ON since 2026-11-15 at the user's request;
// ANVIL_FORGE_TEMPORAL=0 turns it off): the stage keeps
// its own history at INPUT resolution and fuses it BEFORE the net; the net and
// the lanczos3 base then run on the fused frame. Per frame:
//   forge_pyr.comp     luma pyramid of the raw input
//   forge_motion.comp  own block matcher (current vs previous pyramid)
//   forge_fuse.comp    fused = input blended with the motion-compensated
//                      history, limited to +-kappa*T around the input, where
//                      T is the noise gate the net computed on the PREVIOUS
//                      frame; history' = fused (gamma-2 coded)
//   stem, blocks ...   on the fused frame; right after the block that
//                      computes T: forge_gate.comp captures it for the next
//                      frame (first pass creates what the temporal pass needs)
//   forge_tail.comp    unchanged
// Where T == 0 (clean content) and on the first frame the output is
// bit-identical to the spatial path. The gate channel is model-specific:
// "<weights>.gate" (one line "<block> <slot> <scale>", written by
// tools/forge_construct.py) names it; without that file the temporal path
// stays off. History resets on seek and scene cut. Knobs: ANVIL_FORGE_T_KAPPA
// (32), _AMIN (0.25), _TREF (0.004), _HOLD (0.9). Forces one block per
// dispatch. Any input size >= 32x32 (edge pixels past the last full 8x8
// block use the nearest block's motion).
//
// Back-projection (default ON with the temporal path since 2026-11-16;
// ANVIL_FORGE_BP=0 off, =<beta> strength, default 1; ANVIL_FORGE_BP_TREF
// noise reference, default 0.004): shaders/forge_bp.comp after the tail, SDR
// (BT.709 / sRGB) sources only. In the code domain it adds the lanczos3 x3
// upsample of (fused LR - lanczos3 x1/3 of the 4K output), weighted by
// clamp(1 - T/tref) of this frame's stored gate, so the output stays
// consistent with the frame the net saw except where that frame is noise.
// Two passes (LR residual, 4K apply); replica forge_player_io.back_project.
//
// 4K accumulation (default ON with back-projection since 2026-11-16;
// ANVIL_FORGE_ACC=0 off): shaders/forge_acc.comp after the back-projection.
// A history at OUTPUT resolution, warped by its own pyramid Lucas-Kanade
// motion, contributes only what the LR frame cannot see (H - U D H); the
// back-projection step is repeated against it only where the motion brings a
// sub-pixel phase not yet seen (3x3 phase-coverage mask) and the noise gate
// allows. Static content and noise fall back to the single-frame output.
// Replica tools/forge_accum.py (Accum); PLAN §6b.
#pragma once
#include "pipeline/Stages.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace anvil::pipeline {

class ForgeStage : public Stage {
 public:
  static constexpr uint32_t kScale = 3;
  ForgeStage();
  ~ForgeStage() override;

  std::vector<Param> params() override;
  void setInt(const std::string& n, int v) override;

  void prepare(gfx::Gfx& gfx, uint32_t inW, uint32_t inH) override;
  uint32_t outW(uint32_t inW) const override { return active() ? inW * kScale : inW; }
  uint32_t outH(uint32_t inH) const override { return active() ? inH * kScale : inH; }
  void dispatch(gfx::Gfx& gfx, VkCommandBuffer cmd, const gfx::Image& in, gfx::Image& out,
                const FrameMeta& meta) override;

  void onSeek() override { historyValid_ = false; }
  void onSceneCut() override { historyValid_ = false; }

  const std::string& status() const { return status_; }

 private:
  static constexpr uint32_t kStamps = 17;  // max: start, stem, <=12 block groups, tail, bp, acc
  uint32_t fuse() const {  // LDS budget; the temporal capture needs block granularity
    return (storage_ || !stage_ || temporalOn_) ? 1u : uint32_t(fuse_);
  }
  uint32_t nStamps() const { return 3 + 12 / fuse() + (bpOn() ? 1u : 0u) + (accOn() ? 1u : 0u); }
  bool bpOn() const { return temporalOn_ && bpBeta_ > 0.0f; }
  bool accOn() const { return bpOn() && accReq_; }
  bool active() const { return weightsOk_ && pipesReady_ && !tooBig_; }
  bool loadWeights(gfx::Gfx& gfx);
  bool buildPipelines(gfx::Gfx& gfx);
  bool allocFeatures(gfx::Gfx& gfx, uint32_t w, uint32_t h);
  void destroyGpu();
  void recordTiming(const uint64_t* ts);
  void recordAccum(gfx::Gfx& gfx, VkCommandBuffer cmd, gfx::Image& out, uint32_t fi, uint32_t cur,
                   uint32_t prv, bool histOk, uint32_t codeBits, int phase);

  gfx::Gfx* gfx_ = nullptr;
  ComputePass passthru_;  // scaler_comp identity copy when disabled

  bool triedLoad_ = false, weightsOk_ = false;
  std::string status_ = "not loaded";
  gfx::Buffer weights_;
  uint32_t nParams_ = 0;

  int storage_ = 0;  // 0 = fp16 features, 1 = fp32
  int builtStorage_ = -1;
  // Block-kernel tuning knobs (defaults = fastest measured on the 7900 GRE,
  // 2026-11-12; see the F-SPEED record in FORGE-NEW-MODEL.md §6).
  int fuse_ = 1;  // blocks per dispatch (fp16 storage): 1, 2 or 3
  int ppt_ = 1;   // block-kernel pixels per invocation: 1, 2 or 4
  bool stage_ = true;  // block kernel stages tile+halo in shared memory
  bool pipesReady_ = false;
  VkDescriptorSetLayout dsLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline stem_ = VK_NULL_HANDLE, block_ = VK_NULL_HANDLE, tail_ = VK_NULL_HANDLE;
  VkDescriptorSet frameSets_[gfx::kMaxFramesInFlight]{};
  VkDescriptorSet blockSets_[2]{};  // [0]: A -> B, [1]: B -> A

  gfx::Buffer featA_, featB_;
  uint32_t featW_ = 0, featH_ = 0;

  // temporal path
  bool allocTemporal(gfx::Gfx& gfx, uint32_t w, uint32_t h);
  void destroyTemporal();
  bool temporalReq_ = true;   // ANVIL_FORGE_TEMPORAL (0 = spatial only)
  bool gateOk_ = false;       // requested + gate sidecar loaded
  bool temporalOn_ = false;   // gateOk_ + images allocated for the current size
  uint32_t tFailW_ = 0, tFailH_ = 0;  // size at which temporal alloc failed (no retry loop)
  bool tooBig_ = false;       // input larger than maxInPixels_: pass through
  uint64_t maxInPixels_ = 1280ull * 800ull;  // ANVIL_FORGE_MAX_PIXELS (3x of it ~ 4K)
  bool builtTemporal_ = false;
  uint32_t gateBlock_ = 0, gateSlot_ = 0;
  float gateScale_ = 1.0f;
  float kappa_ = 32.0f, amin_ = 0.25f, tref_ = 0.004f, hold_ = 0.9f;
  float ncl_ = 0.0f;  // ANVIL_FORGE_T_NCLAMP: neighbourhood clamp margin, < 0 = off
  float tdead_ = 0.0f;  // ANVIL_FORGE_T_TDEAD: gate dead zone (default set from the dev-set sweep)
  VkPipeline pyrP_ = VK_NULL_HANDLE, motion_ = VK_NULL_HANDLE, fuseP_ = VK_NULL_HANDLE,
             gateP_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkDescriptorSet rawSets_[gfx::kMaxFramesInFlight]{};  // binding 0 = raw input, 4 = fused
  gfx::Image pyr_[2][3];  // [frame parity][level] r16f luma pyramid of the raw input
  gfx::Image gate_[2];    // [frame parity] r16f noise gate (input size)
  gfx::Image hist_[2];    // [frame parity] fused frame, gamma-2 coded (input size)
  gfx::Image flow_, fused_;  // rg16f block motion (input/8); rgba16f fused frame (linear)
  uint32_t frameNo_ = 0;
  bool historyValid_ = false;
  // back-projection (forge_bp.comp)
  float bpBeta_ = 1.0f, bpTref_ = 0.004f;
  VkDescriptorSetLayout bpDsLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout bpLayout_ = VK_NULL_HANDLE;
  VkPipeline bpDown_ = VK_NULL_HANDLE, bpUp_ = VK_NULL_HANDLE;
  VkDescriptorSet bpSets_[gfx::kMaxFramesInFlight]{};
  gfx::Image bpRes_;  // rgba32f LR residual (rgb) + gate weight (a)
  // 4K accumulation (forge_acc.comp)
  bool accReq_ = true;
  float accS0_ = 0.006f, accK_ = 8.0f, accLam_ = 1e-4f;
  VkDescriptorSetLayout accDsLayout_ = VK_NULL_HANDLE;
  VkPipelineLayout accLayout_ = VK_NULL_HANDLE;
  VkPipeline accP_[16]{};  // >= kAccPasses (ForgeStage.cpp)
  VkDescriptorSet accSets_[gfx::kMaxFramesInFlight]{};
  gfx::Image accH_[2];      // [parity] rgba16f history, code domain (output size)
  gfx::Image accE_;         // rgba16f warped history minus this frame (output size)
  gfx::Image lkA_[2][3];    // [parity][level 1..3] r32f LK luma pyramid
  gfx::Image lkF_[4], lkT_[4];  // rg32f LK flow per level (+ pre-smoothing temporary)
  gfx::Image accMask_[2];   // [parity] r32ui 3x3 sub-pixel phase coverage
  gfx::Image accLrA_;       // rg32f (confidence, BP strength)
  gfx::Image accLrD_;       // rgba32f D(E) | (residual, BP strength)
  // ANVIL_FORGE_ACC_PROFILE=1: a timestamp after every accumulation dispatch, logged per pass
  VkQueryPool accQ_ = VK_NULL_HANDLE;
  uint32_t accQn_[gfx::kMaxFramesInFlight]{};
  uint32_t accQpass_[gfx::kMaxFramesInFlight][48]{};
  std::vector<double> accProf_[16];
  uint32_t accProfFrames_ = 0;
  int64_t lastPts_ = INT64_MIN;  // frame fused last (a redraw of it reuses its history)
  bool lastHistOk_ = false;

  // timing
  bool timing_ = false;
  uint32_t timingN_ = 100;
  VkQueryPool qpool_ = VK_NULL_HANDLE;
  bool pending_[gfx::kMaxFramesInFlight]{};
  double tsPeriodNs_ = 1.0;
  std::vector<double> tTotal_, tStem_, tBlocks_, tTail_, tBp_, tAcc_;
};

}  // namespace anvil::pipeline

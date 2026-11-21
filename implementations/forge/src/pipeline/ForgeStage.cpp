#include "pipeline/ForgeStage.hpp"

#include "core/Log.hpp"
#include "core/Paths.hpp"
#include "core/Sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <iterator>

namespace anvil::pipeline {

namespace {

// .forge v1 layout (tools/forge_export.py is the normative description).
constexpr char kMagic[8] = {'F', 'O', 'R', 'G', 'E', 'W', '1', '\0'};
constexpr uint32_t kVersion = 1, kC = 32, kBlocks = 12, kFixedHdr = 68, kEntry = 48;
constexpr uint32_t kTensors = 4 + 4 * kBlocks;
constexpr uint32_t kBlockFloats = kC * 9 + kC + kC * kC + kC;           // 1376
constexpr uint32_t kStemFloats = kC * 27 + kC;                          // 896
constexpr uint32_t kTailOut = 3 * ForgeStage::kScale * ForgeStage::kScale;  // 27
constexpr uint32_t kTotalFloats = kStemFloats + kBlocks * kBlockFloats + kTailOut * kC + kTailOut;
constexpr uint32_t kTailOff = kStemFloats + kBlocks * kBlockFloats;     // 17408

struct Expect {
  std::string name;
  uint32_t ndim;
  uint32_t dims[4];
};

std::vector<Expect> expectedTensors() {
  std::vector<Expect> e;
  e.push_back({"stem.weight", 4, {kC, 3, 3, 3}});
  e.push_back({"stem.bias", 1, {kC, 1, 1, 1}});
  for (uint32_t i = 0; i < kBlocks; i++) {
    const std::string b = "body." + std::to_string(i);
    e.push_back({b + ".dw.weight", 4, {kC, 1, 3, 3}});
    e.push_back({b + ".dw.bias", 1, {kC, 1, 1, 1}});
    e.push_back({b + ".pw.weight", 4, {kC, kC, 1, 1}});
    e.push_back({b + ".pw.bias", 1, {kC, 1, 1, 1}});
  }
  e.push_back({"tail.weight", 4, {kTailOut, kC, 1, 1}});
  e.push_back({"tail.bias", 1, {kTailOut, 1, 1, 1}});
  return e;
}

uint32_t rd32(const std::vector<uint8_t>& b, size_t off) {
  uint32_t v;
  std::memcpy(&v, b.data() + off, 4);  // little-endian host (x86-64)
  return v;
}

struct Push {
  uint32_t inW, inH, outW, outH;
  uint32_t wOff;
  uint32_t flags;  // bit 0: history valid (temporal tail)
  float kappa, amin, tref;
  uint32_t gSlot;
  float gScale;
  float hold;
  float ncl;    // forge_fuse only: neighbourhood clamp margin (< 0 = off)
  float tdead;  // forge_gate only: gate dead zone
};
constexpr uint32_t kBindings = 18;

struct BpPush {  // shaders/forge_bp.comp
  uint32_t inW, inH, outW, outH;
  uint32_t pad;
  uint32_t flags;  // bit 1 BT.709 code domain, bit 2 sRGB
  float beta, tref;
};
constexpr uint32_t kBpBindings = 4;

struct AccPush {  // shaders/forge_acc.comp
  uint32_t inW, inH, outW, outH;
  uint32_t level;
  uint32_t flags;  // bit 1 BT.709 code domain, bit 2 sRGB
  float s0, k, tref, lam;
  float beta;
};
constexpr uint32_t kAccBindings = 27;
constexpr uint32_t kAccPipelineSlots = 16;  // ForgeStage::accP_
enum : uint32_t { kAccPyr, kAccLkInit, kAccLkSolve, kAccLkSmooth, kAccConf, kAccResetMask, kAccWarpE,
                  kAccDownE, kAccHc, kAccDownR, kAccOut, kAccReset, kAccPasses };
static_assert(kAccPasses <= kAccPipelineSlots, "grow ForgeStage::accP_");

void computeBarrier(VkCommandBuffer cmd) {
  VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  di.memoryBarrierCount = 1;
  di.pMemoryBarriers = &mb;
  vkCmdPipelineBarrier2(cmd, &di);
}

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t i = size_t(std::lround(q * double(v.size() - 1)));
  return v[std::min(i, v.size() - 1)];
}

}  // namespace

ForgeStage::ForgeStage() : Stage("forge", "reconstruct") {
  if (const char* s = getenv("ANVIL_FORGE_STORAGE")) storage_ = !strcmp(s, "fp32") ? 1 : 0;
  if (const char* t = getenv("ANVIL_FORGE_TIMING")) timing_ = atoi(t) != 0;
  if (const char* n = getenv("ANVIL_FORGE_TIMING_N")) timingN_ = std::max(1, atoi(n));
  if (const char* f = getenv("ANVIL_FORGE_FUSE")) {
    const int v = atoi(f);
    fuse_ = (v == 1 || v == 2 || v == 3) ? v : 1;  // must divide 12 with an even group count
  }
  if (const char* s = getenv("ANVIL_FORGE_STAGE")) stage_ = atoi(s) != 0;
  if (const char* t = getenv("ANVIL_FORGE_TEMPORAL")) temporalReq_ = atoi(t) != 0;
  if (const char* v = getenv("ANVIL_FORGE_T_KAPPA")) kappa_ = float(atof(v));
  if (const char* v = getenv("ANVIL_FORGE_T_AMIN")) amin_ = float(atof(v));
  if (const char* v = getenv("ANVIL_FORGE_T_TREF")) tref_ = std::max(1e-6f, float(atof(v)));
  if (const char* v = getenv("ANVIL_FORGE_T_HOLD")) hold_ = float(atof(v));
  if (const char* v = getenv("ANVIL_FORGE_T_NCLAMP")) ncl_ = float(atof(v));
  if (const char* v = getenv("ANVIL_FORGE_T_TDEAD")) tdead_ = float(atof(v));
  if (const char* v = getenv("ANVIL_FORGE_BP")) bpBeta_ = std::max(0.0f, float(atof(v)));
  if (const char* v = getenv("ANVIL_FORGE_BP_TREF")) bpTref_ = std::max(1e-6f, float(atof(v)));
  if (const char* v = getenv("ANVIL_FORGE_ACC")) accReq_ = atoi(v) != 0;
  if (const char* v = getenv("ANVIL_FORGE_MAX_PIXELS")) {
    const long long n = atoll(v);
    if (n > 0) maxInPixels_ = uint64_t(n);
  }
  if (const char* p = getenv("ANVIL_FORGE_PPT")) {
    const int v = atoi(p);
    ppt_ = (v == 1 || v == 2 || v == 4) ? v : 1;
  }
}

ForgeStage::~ForgeStage() {
  if (!gfx_ || gfx_->device() == VK_NULL_HANDLE) return;  // device already gone at exit
  vkDeviceWaitIdle(gfx_->device());
  destroyGpu();
  passthru_.shutdown(*gfx_);
  gfx_->destroyBuffer(weights_);
  if (qpool_) vkDestroyQueryPool(gfx_->device(), qpool_, nullptr);
  if (accQ_) vkDestroyQueryPool(gfx_->device(), accQ_, nullptr);
}

std::vector<Param> ForgeStage::params() {
  Param s;
  s.name = "storage";
  s.type = Param::Type::Enum;
  s.enumNames = {"fp16", "fp32"};
  s.ival = storage_;
  return {s};
}

void ForgeStage::setInt(const std::string& n, int v) {
  if (n == "storage") storage_ = v ? 1 : 0;
}

bool ForgeStage::loadWeights(gfx::Gfx& gfx) {
  triedLoad_ = true;
  auto fail = [&](const std::string& why) {
    status_ = why;
    LOG_WARN("forge stage disabled: %s", why.c_str());
    return false;
  };
  std::string path;
  if (const char* env = getenv("ANVIL_FORGE_WEIGHTS"); env && *env) {
    path = env;
  } else {
    std::vector<std::string> cands;
    if (const char* home = getenv("HOME"))
      cands.push_back(std::string(home) + "/.local/share/anvil/forge/default.forge");
    cands.push_back(anvil::paths::repoRoot() + "/assets/forge/merged-v2.forge");
    cands.push_back("assets/forge/merged-v2.forge");
    for (auto& c : cands)
      if (std::ifstream(c, std::ios::binary)) {
        path = c;
        break;
      }
    if (path.empty()) return fail("no weights (ANVIL_FORGE_WEIGHTS unset, assets/forge/merged-v2.forge missing)");
  }
  std::ifstream f(path, std::ios::binary);
  if (!f) return fail("cannot open weights '" + path + "'");
  std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (blob.size() < kFixedHdr) return fail("weights '" + path + "': file too short");
  if (std::memcmp(blob.data(), kMagic, 8) != 0) return fail("weights '" + path + "': bad magic");
  if (rd32(blob, 8) != kVersion) return fail("weights '" + path + "': unsupported version");
  const uint32_t hdrBytes = rd32(blob, 12), nT = rd32(blob, 28), nF = rd32(blob, 32);
  if (rd32(blob, 16) != kC || rd32(blob, 20) != kBlocks || rd32(blob, 24) != kScale ||
      nT != kTensors || nF != kTotalFloats || hdrBytes != kFixedHdr + kEntry * kTensors)
    return fail("weights '" + path + "': architecture mismatch (want C=32 blocks=12 scale=3)");
  if (blob.size() != size_t(hdrBytes) + size_t(nF) * 4)
    return fail("weights '" + path + "': size mismatch (truncated or padded)");
  const auto exp = expectedTensors();
  uint32_t off = 0;
  for (uint32_t i = 0; i < nT; i++) {
    const size_t e = kFixedHdr + size_t(kEntry) * i;
    char name[25] = {};
    std::memcpy(name, blob.data() + e, 24);
    const uint32_t nd = rd32(blob, e + 24);
    bool ok = exp[i].name == name && nd == exp[i].ndim && rd32(blob, e + 44) == off;
    uint32_t cnt = 1;
    for (int d = 0; d < 4; d++) {
      ok = ok && rd32(blob, e + 28 + 4 * d) == exp[i].dims[d];
      cnt *= exp[i].dims[d];
    }
    if (!ok) return fail("weights '" + path + "': tensor table mismatch at '" + exp[i].name + "'");
    off += cnt;
  }
  const uint8_t* payload = blob.data() + hdrBytes;
  const std::string payloadSha = core::sha256Hex(payload, size_t(nF) * 4);
  std::string stored;
  {
    static const char kD[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
      stored += kD[blob[36 + i] >> 4];
      stored += kD[blob[36 + i] & 15];
    }
  }
  if (stored != payloadSha) return fail("weights '" + path + "': payload sha256 mismatch (corrupt)");
  std::vector<float> w(nF);
  std::memcpy(w.data(), payload, size_t(nF) * 4);
  for (float v : w)
    if (!std::isfinite(v)) return fail("weights '" + path + "': non-finite value");

  const VkDeviceSize bytes = VkDeviceSize(nF) * 4;
  if (!gfx.createBuffer(weights_, bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    return fail("weight buffer alloc failed");
  gfx::Buffer staging;
  if (!gfx.createBuffer(staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    gfx.destroyBuffer(weights_);
    return fail("weight staging alloc failed");
  }
  std::memcpy(staging.mapped, w.data(), bytes);
  const bool ran = gfx.immediateSubmit([&](VkCommandBuffer c) {
    VkBufferCopy cp{0, 0, bytes};
    vkCmdCopyBuffer(c, staging.buffer, weights_.buffer, 1, &cp);
  });
  gfx.destroyBuffer(staging);
  if (!ran) {
    gfx.destroyBuffer(weights_);
    return fail("weight upload failed");
  }
  nParams_ = nF;
  weightsOk_ = true;
  if (temporalReq_) {
    // Model-specific: which feature channel is the noise gate, and when.
    std::ifstream g(path + ".gate");
    float sc = 0.0f;
    uint32_t blk = 0, slot = 0;
    if (g && (g >> blk >> slot >> sc) && blk < kBlocks - 1 && slot < kC && sc > 0.0f) {
      gateBlock_ = blk;
      gateSlot_ = slot;
      gateScale_ = sc;
      gateOk_ = true;
      LOG_INFO("forge: temporal path on (gate: block %u slot %u scale %g; kappa=%g amin=%g tref=%g hold=%g nclamp=%g tdead=%g; back-projection %s beta=%g tref=%g)",
               blk, slot, double(sc), double(kappa_), double(amin_), double(tref_), double(hold_), double(ncl_), double(tdead_),
               bpBeta_ > 0.0f ? "on" : "off", double(bpBeta_), double(bpTref_));
      LOG_INFO("forge: 4K accumulation %s", (bpBeta_ > 0.0f && accReq_) ? "on" : "off");
    } else {
      LOG_WARN("forge: temporal path requested but '%s.gate' is missing or invalid — spatial only",
               path.c_str());
    }
  }
  const std::string fileSha = core::sha256Hex(blob.data(), blob.size());
  status_ = "loaded " + path;
  LOG_INFO("forge: weights loaded %s sha256=%s params=%u", path.c_str(),
           fileSha.substr(0, 16).c_str(), nParams_);
  return true;
}

void ForgeStage::destroyGpu() {
  if (!gfx_) return;
  VkDevice dev = gfx_->device();
  destroyTemporal();
  if (sampler_) vkDestroySampler(dev, sampler_, nullptr);
  sampler_ = VK_NULL_HANDLE;
  for (VkPipeline* p : {&stem_, &block_, &tail_, &pyrP_, &motion_, &fuseP_, &gateP_, &bpDown_, &bpUp_}) {
    if (*p) vkDestroyPipeline(dev, *p, nullptr);
    *p = VK_NULL_HANDLE;
  }
  for (auto& s : frameSets_)
    if (s) vkFreeDescriptorSets(dev, gfx_->descPool(), 1, &s), s = VK_NULL_HANDLE;
  for (auto& s : blockSets_)
    if (s) vkFreeDescriptorSets(dev, gfx_->descPool(), 1, &s), s = VK_NULL_HANDLE;
  for (auto& s : rawSets_)
    if (s) vkFreeDescriptorSets(dev, gfx_->descPool(), 1, &s), s = VK_NULL_HANDLE;
  for (auto& s : bpSets_)
    if (s) vkFreeDescriptorSets(dev, gfx_->descPool(), 1, &s), s = VK_NULL_HANDLE;
  for (auto& s : accSets_)
    if (s) vkFreeDescriptorSets(dev, gfx_->descPool(), 1, &s), s = VK_NULL_HANDLE;
  for (auto& p : accP_) {
    if (p) vkDestroyPipeline(dev, p, nullptr);
    p = VK_NULL_HANDLE;
  }
  if (accLayout_) vkDestroyPipelineLayout(dev, accLayout_, nullptr);
  if (accDsLayout_) vkDestroyDescriptorSetLayout(dev, accDsLayout_, nullptr);
  accLayout_ = VK_NULL_HANDLE;
  accDsLayout_ = VK_NULL_HANDLE;
  if (bpLayout_) vkDestroyPipelineLayout(dev, bpLayout_, nullptr);
  if (bpDsLayout_) vkDestroyDescriptorSetLayout(dev, bpDsLayout_, nullptr);
  bpLayout_ = VK_NULL_HANDLE;
  bpDsLayout_ = VK_NULL_HANDLE;
  if (layout_) vkDestroyPipelineLayout(dev, layout_, nullptr);
  if (dsLayout_) vkDestroyDescriptorSetLayout(dev, dsLayout_, nullptr);
  layout_ = VK_NULL_HANDLE;
  dsLayout_ = VK_NULL_HANDLE;
  gfx_->destroyBuffer(featA_);
  gfx_->destroyBuffer(featB_);
  featW_ = featH_ = 0;
  pipesReady_ = false;
  builtStorage_ = -1;
}

bool ForgeStage::buildPipelines(gfx::Gfx& gfx) {
  VkDevice dev = gfx.device();
  // One layout for all kernels:
  //   0 sampled input, 1 weights, 2 feature out, 3 feature in, 4 storage out;
  //   temporal: 5 history (prev), 6 history (write), 7 flow, 8 gate (prev),
  //   9-11 current pyramid (storage), 12-14 previous pyramid (sampled),
  //   15 flow (write), 16 gate (write), 17 feature buffer holding the gate.
  const VkDescriptorType SI = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, SB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         ST = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                         CS = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  VkDescriptorSetLayoutBinding b[kBindings]{};
  const VkDescriptorType types[kBindings] = {SI, SB, SB, SB, ST, CS, ST, CS, CS,
                                             ST, ST, ST, SI, SI, SI, ST, ST, SB};
  for (uint32_t i = 0; i < kBindings; i++)
    b[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dli.bindingCount = kBindings;
  dli.pBindings = b;
  if (vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsLayout_) != VK_SUCCESS) return false;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &dsLayout_;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(dev, &pli, nullptr, &layout_) != VK_SUCCESS) return false;

  struct Spec {
    VkBool32 fp32;
    uint32_t nb, threads, ppt;
    VkBool32 stage;
    VkBool32 temporal;
  } spec{storage_ ? VK_TRUE : VK_FALSE, fuse(), 256u / uint32_t(ppt_), uint32_t(ppt_),
         stage_ ? VK_TRUE : VK_FALSE, temporalOn_ ? VK_TRUE : VK_FALSE};
  VkSpecializationMapEntry me[6] = {{0, offsetof(Spec, fp32), sizeof(VkBool32)},
                                    {1, offsetof(Spec, nb), sizeof(uint32_t)},
                                    {2, offsetof(Spec, threads), sizeof(uint32_t)},
                                    {3, offsetof(Spec, ppt), sizeof(uint32_t)},
                                    {4, offsetof(Spec, stage), sizeof(VkBool32)},
                                    {5, offsetof(Spec, temporal), sizeof(VkBool32)}};
  VkSpecializationInfo si{6, me, sizeof(spec), &spec};
  auto make = [&](const char* name, VkPipeline& out) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = gfx.shader(name);
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = &si;
    ci.layout = layout_;
    return ci.stage.module &&
           vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &out) == VK_SUCCESS;
  };
  if (!make("forge_stem_comp", stem_) || !make("forge_block_comp", block_) ||
      !make("forge_tail_comp", tail_))
    return false;
  if (temporalOn_ &&
      (!make("forge_pyr_comp", pyrP_) || !make("forge_motion_comp", motion_) ||
       !make("forge_fuse_comp", fuseP_) || !make("forge_gate_comp", gateP_)))
    return false;
  {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &sci, nullptr, &sampler_) != VK_SUCCESS) return false;
  }

  if (bpOn()) {
    VkDescriptorSetLayoutBinding bb[kBpBindings]{};
    for (uint32_t i = 0; i < kBpBindings; i++)
      bb[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo bli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    bli.bindingCount = kBpBindings;
    bli.pBindings = bb;
    if (vkCreateDescriptorSetLayout(dev, &bli, nullptr, &bpDsLayout_) != VK_SUCCESS) return false;
    VkPushConstantRange bpr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BpPush)};
    VkPipelineLayoutCreateInfo bpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    bpl.setLayoutCount = 1;
    bpl.pSetLayouts = &bpDsLayout_;
    bpl.pushConstantRangeCount = 1;
    bpl.pPushConstantRanges = &bpr;
    if (vkCreatePipelineLayout(dev, &bpl, nullptr, &bpLayout_) != VK_SUCCESS) return false;
    VkShaderModule mod = gfx.shader("forge_bp_comp");
    if (!mod) return false;
    for (uint32_t pass = 0; pass < 2; pass++) {
      const uint32_t sd[3] = {pass, pass ? 16u : 8u, pass ? 16u : 8u};  // pass 1: one thread per output px
      VkSpecializationMapEntry bme[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
      VkSpecializationInfo bsi{3, bme, sizeof(sd), sd};
      VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      ci.stage.module = mod;
      ci.stage.pName = "main";
      ci.stage.pSpecializationInfo = &bsi;
      ci.layout = bpLayout_;
      if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, pass ? &bpUp_ : &bpDown_) !=
          VK_SUCCESS)
        return false;
    }
    VkDescriptorSetLayout bls[gfx::kMaxFramesInFlight];
    for (auto& l : bls) l = bpDsLayout_;
    VkDescriptorSetAllocateInfo bai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    bai.descriptorPool = gfx.descPool();
    bai.descriptorSetCount = gfx::kMaxFramesInFlight;
    bai.pSetLayouts = bls;
    if (vkAllocateDescriptorSets(dev, &bai, bpSets_) != VK_SUCCESS) return false;
  }

  if (accOn()) {
    VkDescriptorSetLayoutBinding ab[kAccBindings]{};
    for (uint32_t i = 0; i < kAccBindings; i++)
      ab[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ali{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ali.bindingCount = kAccBindings;
    ali.pBindings = ab;
    if (vkCreateDescriptorSetLayout(dev, &ali, nullptr, &accDsLayout_) != VK_SUCCESS) return false;
    VkPushConstantRange apr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AccPush)};
    VkPipelineLayoutCreateInfo apl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    apl.setLayoutCount = 1;
    apl.pSetLayouts = &accDsLayout_;
    apl.pushConstantRangeCount = 1;
    apl.pPushConstantRanges = &apr;
    if (vkCreatePipelineLayout(dev, &apl, nullptr, &accLayout_) != VK_SUCCESS) return false;
    VkShaderModule mod = gfx.shader("forge_acc_comp");
    if (!mod) return false;
    for (uint32_t pass = 0; pass < kAccPasses; pass++) {
      const bool px = pass == kAccHc || pass == kAccOut;  // one thread per output pixel
      const uint32_t sd[3] = {pass, px ? 16u : 8u, px ? 16u : 8u};
      VkSpecializationMapEntry me1[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
      VkSpecializationInfo si1{3, me1, sizeof(sd), sd};
      VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      ci.stage.module = mod;
      ci.stage.pName = "main";
      ci.stage.pSpecializationInfo = &si1;
      ci.layout = accLayout_;
      if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &accP_[pass]) != VK_SUCCESS)
        return false;
    }
    VkDescriptorSetLayout als[gfx::kMaxFramesInFlight];
    for (auto& l : als) l = accDsLayout_;
    VkDescriptorSetAllocateInfo aai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    aai.descriptorPool = gfx.descPool();
    aai.descriptorSetCount = gfx::kMaxFramesInFlight;
    aai.pSetLayouts = als;
    if (vkAllocateDescriptorSets(dev, &aai, accSets_) != VK_SUCCESS) return false;
  }

  VkDescriptorSetLayout ls[4] = {dsLayout_, dsLayout_, dsLayout_, dsLayout_};
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = gfx.descPool();
  ai.descriptorSetCount = gfx::kMaxFramesInFlight;
  ai.pSetLayouts = ls;
  if (vkAllocateDescriptorSets(dev, &ai, frameSets_) != VK_SUCCESS) return false;
  if (vkAllocateDescriptorSets(dev, &ai, rawSets_) != VK_SUCCESS) return false;
  ai.descriptorSetCount = 2;
  if (vkAllocateDescriptorSets(dev, &ai, blockSets_) != VK_SUCCESS) return false;
  builtStorage_ = storage_;
  return true;
}

bool ForgeStage::allocFeatures(gfx::Gfx& gfx, uint32_t w, uint32_t h) {
  gfx.destroyBuffer(featA_);
  gfx.destroyBuffer(featB_);
  const VkDeviceSize bytes = VkDeviceSize(w) * h * (storage_ ? 128 : 64);
  if (!gfx.createBuffer(featA_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
      !gfx.createBuffer(featB_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    return false;
  featW_ = w;
  featH_ = h;
  // Static block sets: weights + (in, out) ping-pong.
  VkDescriptorBufferInfo bw{weights_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo ba{featA_.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo bb{featB_.buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet wr[6]{};
  auto set = [&](int i, VkDescriptorSet s, uint32_t binding, const VkDescriptorBufferInfo* bi) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = s;
    wr[i].dstBinding = binding;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = bi;
  };
  set(0, blockSets_[0], 1, &bw);
  set(1, blockSets_[0], 3, &ba);
  set(2, blockSets_[0], 2, &bb);
  set(3, blockSets_[1], 1, &bw);
  set(4, blockSets_[1], 3, &bb);
  set(5, blockSets_[1], 2, &ba);
  vkUpdateDescriptorSets(gfx.device(), 6, wr, 0, nullptr);
  return true;
}

void ForgeStage::destroyTemporal() {
  if (!gfx_) return;
  for (auto& lv : pyr_)
    for (auto& im : lv) gfx_->destroyImage(im);
  for (auto& im : gate_) gfx_->destroyImage(im);
  for (auto& im : hist_) gfx_->destroyImage(im);
  gfx_->destroyImage(flow_);
  gfx_->destroyImage(fused_);
  gfx_->destroyImage(bpRes_);
  for (auto& im : accH_) gfx_->destroyImage(im);
  gfx_->destroyImage(accE_);
  for (auto& lv : lkA_)
    for (auto& im : lv) gfx_->destroyImage(im);
  for (auto& im : lkF_) gfx_->destroyImage(im);
  for (auto& im : lkT_) gfx_->destroyImage(im);
  for (auto& im : accMask_) gfx_->destroyImage(im);
  gfx_->destroyImage(accLrA_);
  gfx_->destroyImage(accLrD_);
  historyValid_ = false;
  builtTemporal_ = false;
}

bool ForgeStage::allocTemporal(gfx::Gfx& gfx, uint32_t w, uint32_t h) {
  destroyTemporal();
  if (!temporalOn_) return true;
  if (w < 32 || h < 32) {
    LOG_WARN("forge: temporal path needs an input of at least 32x32 (%ux%u) — spatial only", w, h);
    return false;
  }
  const VkImageUsageFlags use = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  bool ok = true;
  for (auto& lv : pyr_)
    for (uint32_t l = 0; l < 3; l++)
      ok = ok && gfx.createImage(lv[l], w >> l, h >> l, VK_FORMAT_R16_SFLOAT, use);
  for (auto& im : gate_) ok = ok && gfx.createImage(im, w, h, VK_FORMAT_R16_SFLOAT, use);
  for (auto& im : hist_) ok = ok && gfx.createImage(im, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, use);
  ok = ok && gfx.createImage(flow_, w / 8, h / 8, VK_FORMAT_R16G16_SFLOAT, use);
  ok = ok && gfx.createImage(fused_, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, use);
  if (bpOn()) ok = ok && gfx.createImage(bpRes_, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, use);
  if (accOn()) {
    const uint32_t W3 = w * kScale, H3 = h * kScale;
    for (auto& im : accH_) ok = ok && gfx.createImage(im, W3, H3, VK_FORMAT_R16G16B16A16_SFLOAT, use);
    ok = ok && gfx.createImage(accE_, W3, H3, VK_FORMAT_R16G16B16A16_SFLOAT, use);
    for (auto& lv : lkA_)
      for (uint32_t l = 1; l < 4; l++)
        ok = ok && gfx.createImage(lv[l - 1], std::max(1u, w >> l), std::max(1u, h >> l), VK_FORMAT_R32_SFLOAT, use);
    for (uint32_t l = 0; l < 4; l++) {
      ok = ok && gfx.createImage(lkF_[l], std::max(1u, w >> l), std::max(1u, h >> l), VK_FORMAT_R32G32_SFLOAT, use);
      ok = ok && gfx.createImage(lkT_[l], std::max(1u, w >> l), std::max(1u, h >> l), VK_FORMAT_R32G32_SFLOAT, use);
    }
    for (auto& im : accMask_) ok = ok && gfx.createImage(im, w, h, VK_FORMAT_R32_UINT, use);
    ok = ok && gfx.createImage(accLrA_, w, h, VK_FORMAT_R32G32_SFLOAT, use);
    ok = ok && gfx.createImage(accLrD_, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, use);
  }
  ok = ok && gfx.immediateSubmit([&](VkCommandBuffer c) {
    for (auto& lv : pyr_)
      for (auto& im : lv) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
    for (auto& im : gate_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
    for (auto& im : hist_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
    gfx.transitionImage(c, flow_, VK_IMAGE_LAYOUT_GENERAL);
    gfx.transitionImage(c, fused_, VK_IMAGE_LAYOUT_GENERAL);
    if (bpOn()) gfx.transitionImage(c, bpRes_, VK_IMAGE_LAYOUT_GENERAL);
    if (accOn()) {
      for (auto& im : accH_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
      gfx.transitionImage(c, accE_, VK_IMAGE_LAYOUT_GENERAL);
      for (auto& lv : lkA_)
        for (auto& im : lv) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
      for (auto& im : lkF_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
      for (auto& im : lkT_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
      for (auto& im : accMask_) gfx.transitionImage(c, im, VK_IMAGE_LAYOUT_GENERAL);
      gfx.transitionImage(c, accLrA_, VK_IMAGE_LAYOUT_GENERAL);
      gfx.transitionImage(c, accLrD_, VK_IMAGE_LAYOUT_GENERAL);
    }
  });
  if (!ok) {
    destroyTemporal();
    return false;
  }
  builtTemporal_ = true;
  return true;
}

void ForgeStage::prepare(gfx::Gfx& gfx, uint32_t inW, uint32_t inH) {
  gfx_ = &gfx;
  if (!triedLoad_) loadWeights(gfx);
  const bool big = uint64_t(inW) * uint64_t(inH) > maxInPixels_;
  if (big && !tooBig_ && weightsOk_)
    LOG_INFO("forge: %ux%u input is above the %llu-pixel cap — passing through", inW, inH,
             (unsigned long long)maxInPixels_);
  tooBig_ = big;
  if (!weightsOk_ || tooBig_) {
    if (!passthru_.valid()) passthru_.init(gfx, "scaler_comp");
    return;
  }
  const bool wantT = gateOk_ && inW >= 32 && inH >= 32 && !(inW == tFailW_ && inH == tFailH_);
  if (pipesReady_ && builtStorage_ == storage_ && featW_ == inW && featH_ == inH &&
      temporalOn_ == wantT)
    return;
  vkDeviceWaitIdle(gfx.device());  // feature buffers may be in flight
  if (builtStorage_ != storage_ || temporalOn_ != wantT) {
    temporalOn_ = wantT;  // the kernels are specialised on it
    destroyGpu();
    if (!buildPipelines(gfx)) {
      destroyGpu();
      status_ = "pipeline creation failed";
      LOG_WARN("forge stage disabled: %s", status_.c_str());
      weightsOk_ = false;
      passthru_.init(gfx, "scaler_comp");
      return;
    }
  }
  if (!allocFeatures(gfx, inW, inH)) {
    status_ = "feature buffer alloc failed";
    LOG_WARN("forge stage disabled: %s", status_.c_str());
    destroyGpu();
    weightsOk_ = false;
    passthru_.init(gfx, "scaler_comp");
    return;
  }
  if (temporalOn_ && !allocTemporal(gfx, inW, inH)) {
    // Fall back to the spatial path: rebuild the kernels without TEMPORAL.
    temporalOn_ = false;
    tFailW_ = inW;
    tFailH_ = inH;
    destroyGpu();
    if (!buildPipelines(gfx) || !allocFeatures(gfx, inW, inH)) {
      destroyGpu();
      status_ = "pipeline creation failed";
      LOG_WARN("forge stage disabled: %s", status_.c_str());
      weightsOk_ = false;
      passthru_.init(gfx, "scaler_comp");
      return;
    }
  }
  pipesReady_ = true;
  LOG_INFO("forge: prepared %ux%u -> %ux%u, %s feature storage (%.1f MB x2), %u block(s)/dispatch",
           inW, inH, inW * kScale, inH * kScale, storage_ ? "fp32" : "fp16",
           double(featA_.size) / (1024.0 * 1024.0), fuse());
}

void ForgeStage::recordTiming(const uint64_t* ts) {
  auto ms = [&](uint32_t a, uint32_t b) { return double(ts[b] - ts[a]) * tsPeriodNs_ * 1e-6; };
  const uint32_t n = nStamps();
  for (uint32_t i = 1; i < n; i++)
    if (ts[i] < ts[i - 1]) return;  // wrapped / invalid
  tTotal_.push_back(ms(0, n - 1));
  tStem_.push_back(ms(0, 1));
  tBlocks_.push_back(ms(1, n - 2));
  const uint32_t tl = n - 1 - (bpOn() ? 1u : 0u) - (accOn() ? 1u : 0u);  // stamp after the tail
  tBlocks_.back() = ms(1, tl - 1);
  tTail_.push_back(ms(tl - 1, tl));
  tAcc_.push_back(accOn() ? ms(tl, tl + 1) : 0.0);              // acc phase 0 (LK, confidence)
  tBp_.push_back(bpOn() ? ms(accOn() ? tl + 1 : tl, n - 1) : 0.0);  // BP (+ acc 4K passes)
  if (tTotal_.size() >= timingN_) {
    LOG_INFO("forge: gpu ms median=%.3f p90=%.3f over %zu frames (%s storage, fuse=%u ppt=%d "
             "lds=%d; median stem=%.3f blocks=%.3f [%.3f/block] tail=%.3f bp+acc4k=%.3f acc-lk=%.3f)",
             pct(tTotal_, 0.5), pct(tTotal_, 0.9), tTotal_.size(), storage_ ? "fp32" : "fp16",
             fuse(), ppt_, int(stage_), pct(tStem_, 0.5), pct(tBlocks_, 0.5),
             pct(tBlocks_, 0.5) / kBlocks, pct(tTail_, 0.5), pct(tBp_, 0.5), pct(tAcc_, 0.5));
    tTotal_.clear();
    tStem_.clear();
    tBlocks_.clear();
    tTail_.clear();
    tBp_.clear();
    tAcc_.clear();
  }
}

// phase 0: before the back-projection (LK pyramid + flow + confidence/phase mask; on a reset
// frame the mask reset); phase 1: after the back-projection on a history frame (warp, history
// passes); phase 2: after the back-projection on a reset frame (history = this frame).
void ForgeStage::recordAccum(gfx::Gfx& gfx, VkCommandBuffer cmd, gfx::Image& out, uint32_t fi,
                             uint32_t cur, uint32_t prv, bool histOk, uint32_t codeBits, int phase) {
  const uint32_t W = featW_, H = featH_, W3 = W * kScale, H3 = H * kScale;
  const VkImageLayout GEN = VK_IMAGE_LAYOUT_GENERAL;
  const gfx::Image* img[kAccBindings] = {
      &out,          &accH_[prv],   &accH_[cur],   &accE_,        &fused_,       &gate_[cur],
      &pyr_[cur][0], &pyr_[prv][0], &lkA_[cur][0], &lkA_[cur][1], &lkA_[cur][2], &lkA_[prv][0],
      &lkA_[prv][1], &lkA_[prv][2], &lkF_[0],      &lkF_[1],      &lkF_[2],      &lkF_[3],
      &lkT_[0],      &lkT_[1],      &lkT_[2],      &lkT_[3],      &accMask_[prv], &accMask_[cur],
      &accLrA_,      &accLrD_,      &bpRes_};
  VkDescriptorImageInfo ii[kAccBindings];
  VkWriteDescriptorSet wr[kAccBindings]{};
  for (uint32_t b = 0; b < kAccBindings; b++) {
    ii[b] = {VK_NULL_HANDLE, img[b]->view, GEN};
    wr[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[b].dstSet = accSets_[fi];
    wr[b].dstBinding = b;
    wr[b].descriptorCount = 1;
    wr[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[b].pImageInfo = &ii[b];
  }
  if (phase == 0) vkUpdateDescriptorSets(gfx.device(), kAccBindings, wr, 0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accLayout_, 0, 1, &accSets_[fi], 0, nullptr);
  static const bool kDbg = getenv("ANVIL_FORGE_ACC_DEBUG") != nullptr;  // flow/confidence view
  static const bool kDbgFine = kDbg && std::string(getenv("ANVIL_FORGE_ACC_DEBUG")) == "fine";
  AccPush ap{W, H, W3, H3, 0, codeBits | (kDbg ? 256u : 0u) | (kDbgFine ? 512u : 0u), accS0_, accK_, bpTref_, accLam_,
             bpBeta_};
  static const bool kProf = getenv("ANVIL_FORGE_ACC_PROFILE") != nullptr;
  if (kProf && !accQ_) {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = 49 * gfx::kMaxFramesInFlight;
    vkCreateQueryPool(gfx.device(), &qi, nullptr, &accQ_);
  }
  if (kProf && accQ_ && phase == 0) {
    const uint32_t n = accQn_[fi];
    if (n > 1) {  // this slot's previous frame has completed (fence)
      uint64_t ts[49];
      if (vkGetQueryPoolResults(gfx.device(), accQ_, fi * 49, n, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT) ==
          VK_SUCCESS) {
        double per[16] = {};
        for (uint32_t i = 1; i < n; i++)
          per[accQpass_[fi][i]] += double(ts[i] - ts[i - 1]) * double(gfx.props.limits.timestampPeriod) * 1e-6;
        for (int k = 0; k < 16; k++) accProf_[k].push_back(per[k]);
        if (++accProfFrames_ >= 100) {
          static const char* nm[16] = {"pyr", "lkinit", "lksolve", "lksmooth", "conf", "resetmask", "warpE",
                                       "downE", "hc", "downR", "out", "reset", "bp", "-", "-", "-"};
          std::string line;
          for (int k = 0; k < 16; k++)
            if (pct(accProf_[k], 0.5) > 0.0) line += std::string(" ") + nm[k] + "=" + std::to_string(pct(accProf_[k], 0.5)).substr(0, 5);
          LOG_INFO("forge: acc profile median ms:%s", line.c_str());
          for (auto& v : accProf_) v.clear();
          accProfFrames_ = 0;
        }
      }
    }
    vkCmdResetQueryPool(cmd, accQ_, fi * 49, 49);
    accQn_[fi] = 0;
  }
  auto tsAcc = [&](uint32_t pass) {
    if (!kProf || !accQ_ || accQn_[fi] >= 49) return;
    accQpass_[fi][accQn_[fi]] = pass;
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, accQ_, fi * 49 + accQn_[fi]++);
  };
  tsAcc(phase == 0 ? 0u : 12u);  // phase 1/2: the interval since phase 0 is the back-projection
  auto run = [&](uint32_t pass, uint32_t level, uint32_t gx, uint32_t gy) {
    computeBarrier(cmd);
    ap.level = level;
    vkCmdPushConstants(cmd, accLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ap), &ap);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, accP_[pass]);
    vkCmdDispatch(cmd, gx, gy, 1);
    tsAcc(pass);
  };
  auto lv = [](uint32_t n, uint32_t l) { return (std::max(1u, n >> l) + 7) / 8; };
  // profiling only: ANVIL_FORGE_ACC_SKIP bitmask drops pass groups (1 LK, 2 warp, 4 downE,
  // 8 Hc, 16 downR, 32 out, 64 conf) — the output is wrong when set
  static const uint32_t kSkip = [] { const char* e = getenv("ANVIL_FORGE_ACC_SKIP"); return e ? uint32_t(atoi(e)) : 0u; }();
  auto run0 = run;
  auto runIf = [&](uint32_t bit, uint32_t pass, uint32_t level, uint32_t gx, uint32_t gy) {
    if (!(kSkip & bit)) run0(pass, level, gx, gy);
  };
  if (phase == 0) {
    run(kAccPyr, 0, lv(W, 1), lv(H, 1));
    if (histOk) {
      // per level: 3 x (solve -> T, 5x5 smooth T -> F); the first solve starts from the
      // level init (coarser flow, bilinear x2, times 2) instead of F
      for (int l = 3; l >= 0; l--) {
        const uint32_t L = uint32_t(l);
        for (int it = 0; it < 3; it++) {
          runIf(1, kAccLkSolve, L | (it == 0 ? 1u << 8 : 0u), lv(W, L), lv(H, L));
          runIf(1, kAccLkSmooth, L, lv(W, L), lv(H, L));
        }
      }
      runIf(64, kAccConf, 0, (W + 7) / 8, (H + 7) / 8);
    } else {
      run(kAccResetMask, 0, (W + 7) / 8, (H + 7) / 8);
    }
  } else if (phase == 1) {
    runIf(2, kAccWarpE, 0, (W3 + 7) / 8, (H3 + 7) / 8);
    runIf(4, kAccDownE, 0, (W + 7) / 8, (H + 7) / 8);
    runIf(8, kAccHc, 0, (W3 + 15) / 16, (H3 + 15) / 16);
    runIf(16, kAccDownR, 0, (W + 7) / 8, (H + 7) / 8);
    runIf(32, kAccOut, 0, (W3 + 15) / 16, (H3 + 15) / 16);
  } else {
    run(kAccReset, 0, (W3 + 7) / 8, (H3 + 7) / 8);
  }
}

void ForgeStage::dispatch(gfx::Gfx& gfx, VkCommandBuffer cmd, const gfx::Image& in,
                          gfx::Image& out, const FrameMeta& meta) {
  const uint32_t fi = gfx.frameIndex();
  if (weightsOk_ && builtStorage_ != storage_) prepare(gfx, in.extent.width, in.extent.height);
  const uint32_t W = in.extent.width, H = in.extent.height;
  if (!active() || out.extent.width != W * kScale || out.extent.height != H * kScale ||
      featW_ != W || featH_ != H) {
    // Disabled: identity copy (scaler.comp nearest at 1:1 is an exact passthrough).
    if (!passthru_.valid() && !passthru_.init(gfx, "scaler_comp")) return;
    passthru_.bind(cmd, gfx, fi, in, out);
    ComputePass::Push p{};
    p.inW = W;
    p.inH = H;
    p.outW = out.extent.width;
    p.outH = out.extent.height;
    p.mode = 3;
    passthru_.push(cmd, p);
    passthru_.dispatch(cmd, (out.extent.width + 7) / 8, (out.extent.height + 7) / 8);
    return;
  }

  if (timing_ && !qpool_) {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = kStamps * gfx::kMaxFramesInFlight;
    if (vkCreateQueryPool(gfx.device(), &qi, nullptr, &qpool_) != VK_SUCCESS) {
      LOG_WARN("forge: timestamp pool create failed — timing off");
      timing_ = false;
    }
    tsPeriodNs_ = gfx.props.limits.timestampPeriod;
  }
  if (qpool_) {
    // This frame slot's previous submission has completed (beginFrame waited
    // on its fence), so its stamps are final.
    if (pending_[fi]) {
      uint64_t ts[kStamps];
      if (vkGetQueryPoolResults(gfx.device(), qpool_, fi * kStamps, nStamps(), nStamps() * 8, ts,
                                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        recordTiming(ts);
      pending_[fi] = false;
    }
    vkCmdResetQueryPool(cmd, qpool_, fi * kStamps, kStamps);
  }
  auto stamp = [&](uint32_t i) {
    if (qpool_)
      vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, qpool_, fi * kStamps + i);
  };

  // Per-frame set: input texture, weights, stem out (A), tail in (A), output.
  // Temporal: the net's set reads the FUSED frame at binding 0; a second set
  // (rawSets_) carries the raw input and the temporal images for the
  // pyramid / motion / fusion / gate kernels.
  const bool temporal = temporalOn_ && builtTemporal_;
  // A redraw of the frame we fused last (pause, capture) must not advance the
  // history: redo that frame against the same previous state.
  const bool repeat = temporal && historyValid_ && frameNo_ > 0 && meta.id.pts_ticks == lastPts_ &&
                      !meta.sceneCut && !meta.firstAfterSeek;
  if (repeat) frameNo_--;
  const uint32_t cur = frameNo_ & 1u, prv = cur ^ 1u;
  const bool histOk = repeat ? lastHistOk_
                             : (temporal && historyValid_ && !meta.sceneCut && !meta.firstAfterSeek);
  {
    const VkImageLayout GEN = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo din{VK_NULL_HANDLE, in.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dfused{VK_NULL_HANDLE, fused_.view, GEN};
    VkDescriptorImageInfo dout{VK_NULL_HANDLE, out.view, GEN};
    VkDescriptorBufferInfo bw{weights_.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ba{featA_.buffer, 0, VK_WHOLE_SIZE};
    // the gate block writes B when its index is even (stem -> A, block 0 -> B, ...)
    VkDescriptorBufferInfo bg{(gateBlock_ & 1u) ? featA_.buffer : featB_.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorType SI = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, SB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           ST = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                           CS = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    const VkDescriptorType types[kBindings] = {SI, SB, SB, SB, ST, CS, ST, CS, CS,
                                               ST, ST, ST, SI, SI, SI, ST, ST, SB};
    VkWriteDescriptorSet wr[2 * kBindings]{};
    uint32_t n = 0;
    auto add = [&](VkDescriptorSet set, uint32_t binding, const VkDescriptorImageInfo* ii,
                   const VkDescriptorBufferInfo* bi) {
      wr[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wr[n].dstSet = set;
      wr[n].dstBinding = binding;
      wr[n].descriptorCount = 1;
      wr[n].descriptorType = types[binding];
      wr[n].pImageInfo = ii;
      wr[n].pBufferInfo = bi;
      n++;
    };
    add(frameSets_[fi], 0, temporal ? &dfused : &din, nullptr);
    add(frameSets_[fi], 1, nullptr, &bw);
    add(frameSets_[fi], 2, nullptr, &ba);
    add(frameSets_[fi], 3, nullptr, &ba);  // an even number of block groups ends back in A
    add(frameSets_[fi], 4, &dout, nullptr);
    VkDescriptorImageInfo ii[kBindings]{};
    if (temporal) {
      auto img = [&](const gfx::Image& t, bool smp) {
        return VkDescriptorImageInfo{smp ? sampler_ : VK_NULL_HANDLE, t.view, GEN};
      };
      ii[0] = din;
      ii[4] = img(fused_, false);
      ii[5] = img(hist_[prv], true);
      ii[6] = img(hist_[cur], false);
      ii[7] = img(flow_, true);
      ii[8] = img(gate_[prv], true);
      for (uint32_t l = 0; l < 3; l++) {
        ii[9 + l] = img(pyr_[cur][l], false);
        ii[12 + l] = img(pyr_[prv][l], false);
      }
      ii[15] = img(flow_, false);
      ii[16] = img(gate_[cur], false);
      for (uint32_t b : {0u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
        add(rawSets_[fi], b, &ii[b], nullptr);
      add(rawSets_[fi], 17, nullptr, &bg);
    }
    vkUpdateDescriptorSets(gfx.device(), n, wr, 0, nullptr);
  }

  // Shared feature buffers (and temporal images): order against any earlier
  // forge work on the queue (the previous frame in flight may still be
  // reading them).
  computeBarrier(cmd);
  stamp(0);
  // Base domain: content is downscaled on its code values (swscale and
  // practically every real 720p), so the lanczos3 base is taken there too —
  // BT.709 OETF (transfer 3) or sRGB (0) — and decoded back to linear.
  // PQ/HLG keep the linear-light base. ANVIL_FORGE_BASE=linear forces it.
  static const bool kLinearBase = [] {
    const char* e = getenv("ANVIL_FORGE_BASE");
    return e != nullptr && std::string(e) == "linear";
  }();
  uint32_t baseBits = 0;
  if (!kLinearBase) baseBits = meta.srcColor.transfer == 3 ? 2u : (meta.srcColor.transfer == 0 ? 4u : 0u);
  Push pc{W, H, W * kScale, H * kScale, 0, (histOk ? 1u : 0u) | baseBits, kappa_, amin_, tref_,
          gateSlot_, gateScale_, hold_, ncl_, tdead_};
  auto pushIt = [&]() {
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
  };
  auto bindSet = [&](VkDescriptorSet* s) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, s, 0, nullptr);
  };

  if (temporal) {
    // pyramid -> motion -> fusion, all at input resolution, before the net
    bindSet(&rawSets_[fi]);
    pushIt();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pyrP_);
    vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
    if (histOk) {
      computeBarrier(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, motion_);
      vkCmdDispatch(cmd, (W / 8 + 7) / 8, (H / 8 + 7) / 8, 1);
    }
    computeBarrier(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fuseP_);
    vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
    computeBarrier(cmd);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, stem_);
  bindSet(&frameSets_[fi]);
  pc.wOff = 0;
  pushIt();
  vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);
  stamp(1);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, block_);
  const uint32_t nb = fuse(), groups = kBlocks / nb;  // groups is even: result ends in A
  for (uint32_t i = 0; i < groups; i++) {
    computeBarrier(cmd);
    bindSet(&blockSets_[i & 1]);
    pc.wOff = kStemFloats + i * nb * kBlockFloats;
    pushIt();
    vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);
    if (temporal && i == gateBlock_) {
      // First-pass capture for the next frame: the gate lives in the buffer
      // this block just wrote, and the next-but-one block overwrites it.
      computeBarrier(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gateP_);
      bindSet(&rawSets_[fi]);
      vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, block_);
    }
    stamp(2 + i);
  }

  computeBarrier(cmd);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tail_);
  bindSet(&frameSets_[fi]);
  pc.wOff = kTailOff;
  pushIt();
  vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
  // stamps after the tail: [acc phase 0 (LK, confidence)] then [BP + acc 4K passes]
  stamp(nStamps() - 1 - (bpOn() ? 1u : 0u) - (accOn() ? 1u : 0u));
  const uint32_t tr = meta.srcColor.transfer;
  const bool sdr = temporal && (tr == 3 || tr == 0);
  const uint32_t codeBits = tr == 0 ? 4u : 2u;
  if (accOn()) {
    if (sdr) recordAccum(gfx, cmd, out, fi, cur, prv, histOk, codeBits, 0);
    stamp(nStamps() - 2);
  }
  if (bpOn()) {
    if (sdr) {
      const VkImageLayout GEN = VK_IMAGE_LAYOUT_GENERAL;
      VkDescriptorImageInfo ii[kBpBindings] = {{VK_NULL_HANDLE, fused_.view, GEN},
                                               {VK_NULL_HANDLE, gate_[cur].view, GEN},
                                               {VK_NULL_HANDLE, out.view, GEN},
                                               {VK_NULL_HANDLE, bpRes_.view, GEN}};
      VkWriteDescriptorSet wr[kBpBindings]{};
      for (uint32_t b = 0; b < kBpBindings; b++) {
        wr[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[b].dstSet = bpSets_[fi];
        wr[b].dstBinding = b;
        wr[b].descriptorCount = 1;
        wr[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[b].pImageInfo = &ii[b];
      }
      vkUpdateDescriptorSets(gfx.device(), kBpBindings, wr, 0, nullptr);
      BpPush bp{W, H, W * kScale, H * kScale, 0, codeBits, bpBeta_, bpTref_};
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bpLayout_, 0, 1, &bpSets_[fi], 0,
                              nullptr);
      vkCmdPushConstants(cmd, bpLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp), &bp);
      computeBarrier(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bpDown_);
      vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
      computeBarrier(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bpUp_);
      vkCmdDispatch(cmd, (W * kScale + 15) / 16, (H * kScale + 15) / 16, 1);  // one thread per output px
      if (accOn()) recordAccum(gfx, cmd, out, fi, cur, prv, histOk, codeBits, histOk ? 1 : 2);
    }
    stamp(nStamps() - 1);
  }
  if (qpool_) pending_[fi] = true;
  if (temporal) {
    static const bool kLog = getenv("ANVIL_FORGE_T_LOG") != nullptr;
    if (kLog)
      LOG_INFO("forge: temporal frame %u pts_ticks=%lld history=%d cut=%d seek=%d", frameNo_, (long long)meta.id.pts_ticks,
               int(histOk), int(meta.sceneCut), int(meta.firstAfterSeek));
    historyValid_ = true;
    lastPts_ = meta.id.pts_ticks;
    lastHistOk_ = histOk;
    frameNo_++;
  }
}

}  // namespace anvil::pipeline

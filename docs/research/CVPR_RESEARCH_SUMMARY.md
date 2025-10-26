# CVPR 2026 Research Proposal - Executive Summary

> **Quick Reference Guide** | Full LaTeX version: [CVPR_RESEARCH_PROPOSAL.tex](CVPR_RESEARCH_PROPOSAL.tex)
>
> **Target:** CVPR 2026 (Submission: ~November 2025)
>
> **Status:** Ready to implement
>
> **Created:** October 26, 2025

---

## Current Achievements

### Platform
- **Hardware:** NVIDIA RTX 5080 (16GB VRAM, Blackwell sm_120)
- **Software:** PyTorch 2.9, CUDA 12.8, Taming 3DGS fully functional

### Training Results
- **Budget Mode:** 13/13 datasets successful (2h 2m, avg PSNR 25.93 dB)
- **Big Mode:** 13/13 datasets (18.8M Gaussians, 27.66 GB)
- **Memory Efficiency:** 3× reduction (48GB → 16GB VRAM)

### Technical Contributions Already Made
1. Blackwell architecture support
2. Memory optimizations (CPU data device, expandable segments)
3. Adaptive densification intervals
4. Bug fixes in densification algorithm

---

## Six Proposed Research Directions

### 🔥 Direction 1: Memory-Efficient 3DGS for Consumer Hardware **[HIGH PRIORITY]**

**Problem:** Original 3DGS requires expensive 48GB GPUs ($4,500+ A6000)

**Solution:** **AdaptiveGS** - Dynamic VRAM-aware training

**Key Innovation:**
- Real-time VRAM monitoring
- Adaptive densification interval scheduling
- Hierarchical Gaussian representation (LOD-like)

**Expected Results:**
- Train on 8GB GPUs (RTX 4060 Ti)
- <5% quality loss vs 48GB baseline
- 10× hardware cost reduction

**Timeline:** 10 weeks

**CVPR Fit:** Good (democratization angle)

---

### 🔥🔥 Direction 2: Neural Importance Networks **[HIGHEST NOVELTY]**

**Problem:** Current importance scoring uses 12 hand-tuned coefficients that are scene-dependent and non-optimal

**Solution:** **LearnedDensify** - Replace hand-crafted scoring with small MLP

**Architecture:**
```
Input: [gradient, opacity, depth, radii, scale,
        dist_accum, loss_accum, blend, count] (13D)
↓
3-layer MLP (64 hidden dims, ~5K parameters)
↓
Output: Importance score (1D)
```

**Training Strategy:**
- Meta-learning (MAML/Reptile) across multiple scenes
- Optimize for final reconstruction quality
- Gumbel-Softmax for gradient flow through sampling

**Key Innovation:**
- First learned importance function for 3DGS
- Differentiable densification
- Zero-shot generalization to new scene types

**Expected Results:**
- +1-2 dB PSNR improvement
- Faster convergence (20k vs 30k iterations)
- Learns scene-specific importance weights

**Timeline:** 12 weeks

**CVPR Fit:** **Excellent** (highest novelty, aligns with learned optimization trend)

**🎯 RECOMMENDED AS PRIMARY DIRECTION**

---

### Direction 3: Progressive Multi-Resolution Training **[EFFICIENCY]**

**Problem:** Fixed resolution throughout training is wasteful

**Solution:** **ProgressiveGS** - Coarse-to-fine training schedule

**Schedule Example:**
```
Iterations 0-5000:    images_8  (12.5% resolution)
Iterations 5000-10000: images_4  (25%)
Iterations 10000-15000: images_2  (50%)
Iterations 15000-30000: images    (100%)
```

**Expected Results:**
- 2-3× training speedup
- Same or better final quality
- Reduced early-stage VRAM usage

**Timeline:** 8 weeks

**CVPR Fit:** Good (efficiency focus)

---

### Direction 4: Uncertainty-Aware 3DGS **[ROBUSTNESS]**

**Problem:** No uncertainty quantification - cannot assess reconstruction confidence

**Solution:** **UncertaintyGS** - Probabilistic Gaussians

**Method:**
- Extend each Gaussian to encode parameter uncertainty:
  - Position: μ_pos ~ N(x, Σ_pos)
  - Opacity: α ~ Beta(a, b)
  - Scale: s ~ LogNormal(μ_s, σ_s)
- Sampling-based rendering (K=5 samples)
- Calibration loss for reliable uncertainty

**Applications:**
- Novel view quality assessment
- Active view selection (where to capture next?)
- Safety-critical applications

**Expected Results:**
- Uncertainty correlates with error (Pearson r > 0.7)
- Active learning reduces required views by 30-40%
- <20% computational overhead

**Timeline:** 10 weeks

**CVPR Fit:** **Excellent** (uncertainty in 3D is trending)

**🎯 RECOMMENDED AS SECONDARY DIRECTION** (pairs well with LearnedDensify)

---

### Direction 5: Compression and Pruning **[DEPLOYMENT]**

**Problem:** Big mode uses 2.1GB per scene - too large for mobile/streaming

**Solution:** **CompressedGS** - Multi-stage compression pipeline

**Compression Stages:**
1. **Pruning:** Remove low-opacity/rarely-visible Gaussians (30-50% reduction)
2. **Quantization:** Float32→Float16, learned codebooks (4-8× reduction)
3. **Entropy Coding:** Arithmetic coding (1.5-2× reduction)

**Overall:** 16.7× compression → 2.1GB → **125MB per scene**

**Expected Results:**
- 10-20× compression
- <1 dB PSNR loss
- Real-time decompression (>30 FPS)

**Timeline:** 10 weeks

**CVPR Fit:** Good (deployment focus)

---

### Direction 6: Dynamic Scene Reconstruction **[EXTENSION]**

**Problem:** Current 3DGS limited to static scenes

**Solution:** **DynamicGS** - Time-dependent Gaussians with deformation fields

**Method:**
```
x(t) = x_0 + D_pos(x_0, t; θ)
s(t) = s_0 · exp(D_scale(x_0, t; θ))
q(t) = normalize(q_0 + D_rot(x_0, t; θ))
α(t) = σ(α_0 + D_opac(x_0, t; θ))
```
Where D are hash-encoded MLPs (Instant-NGP style)

**Key Innovation:**
- Memory-efficient temporal encoding (hash grids)
- Shared deformation networks
- Temporal smoothness regularization

**Expected Results:**
- Match D-3DGS quality with 2-3× less memory
- Train on RTX 5080 (vs 48GB requirement)
- Real-time dynamic rendering

**Timeline:** 12 weeks

**CVPR Fit:** **Excellent** (new capability, high impact)

---

## Recommended Strategy

### 🏆 **Option A: Single Strong Paper (RECOMMENDED)**

**Focus:** Direction 2 (LearnedDensify) + Direction 4 (UncertaintyGS)

**Title:** "Learning to Densify: Neural Importance Networks for Uncertainty-Aware 3D Gaussian Splatting"

**Rationale:**
- Highest novelty (first learned densification for 3DGS)
- Natural synergy (uncertainty guides densification)
- Strong narrative arc (hand-crafted → learned)
- Clear baseline (Taming 3DGS hand-tuned coefficients)

**Timeline:** 20-24 weeks (5-6 months) ✅ **Fits CVPR 2026 deadline!**

**Expected Results:**
| Metric | Taming 3DGS | LearnedDensify (ours) | Gain |
|--------|-------------|----------------------|------|
| PSNR | 25.93 dB | 27.1 dB | +1.2 dB |
| SSIM | 0.762 | 0.791 | +0.03 |
| LPIPS | 0.223 | 0.198 | -0.025 |

**Minimum Viable Contribution:**
- +0.5 dB PSNR **OR**
- 2× speedup **OR**
- Novel uncertainty capability (first for 3DGS)

---

### Option B: Multiple Papers

**Main:** Direction 2 (LearnedDensify)
**Workshop 1:** Direction 1 (AdaptiveGS) - "Efficient Deep Learning" workshop
**Workshop 2:** Direction 5 (CompressedGS) - "Neural Compression" workshop

**Pro:** Maximize publication count
**Con:** Dilutes focus

---

### Option C: High-Risk High-Reward

**Focus:** Direction 6 (DynamicGS) only

**Pro:** Huge potential impact (new capability)
**Con:** Harder to implement, requires new datasets

---

## Detailed Implementation: LearnedDensify

### Week-by-Week Plan

**Week 1 (Now):**
- Set up branch: `feature/learned-importance`
- Implement basic ImportanceNet (3-layer MLP, 64 hidden)
- Integrate forward pass into training loop
- Sanity check: Does network predict reasonable scores?

**Week 2-3:**
- Implement gradient flow (Gumbel-Softmax estimator)
- Train on single scene (bicycle) with fixed LR
- Compare learned vs hand-crafted importance
- **Go/No-Go Decision Point**

**Week 4-6:**
- Implement MAML/Reptile meta-learning
- Train on 10 scenes, validate on 3 held-out
- Extensive ablations (network size, features, etc.)

**Week 7-9:**
- Add uncertainty extension (probabilistic Gaussians)
- Integrate calibration losses
- Test active view selection

**Week 10-12:**
- Full benchmark on all 13 datasets
- Ablation studies (Table 1-4 in paper)
- Generate qualitative comparisons

**Week 13-16:**
- Paper writing (8 pages + supplementary)
- Video results, code cleanup
- Final experiments based on reviews

**Week 17-20:**
- Iterate on paper based on feedback
- Prepare rebuttal materials
- Submit to CVPR 2026 (mid-November)

---

## Resource Requirements

### Hardware (Current)
✅ NVIDIA RTX 5080 (16GB) - Already have
✅ Modern CPU, 64GB+ RAM - Already have
✅ Storage for datasets - Already have

### Hardware (Recommended Additional)
- RTX 5090 (32GB) for big mode experiments - $2,000
- RTX 4090 for parallel training - $1,800
- 4TB NVMe SSD - $300

**Total Additional:** $4,100 (optional, not critical)

### Software (All Free)
✅ Taming 3DGS codebase - Already have
✅ PyTorch 2.9, CUDA 12.8 - Already have
✅ MipNeRF360, Tanks&Temples, DeepBlending - Already downloaded
- Meta-learning libs: learn2learn, higher - Free
- For DynamicGS: D-NeRF dataset - Free download

---

## Risk Mitigation

### Technical Risks

**Risk 1:** Learned network doesn't improve over hand-crafted

**Mitigation:**
- Start simple (linear model), increase complexity gradually
- Even matching hand-crafted with automation is publishable
- Fall back to per-scene learning (still novel)

**Risk 2:** Meta-learning unstable or doesn't generalize

**Mitigation:**
- Single-scene learned importance still valid contribution
- Try multiple meta-learning algorithms (MAML, Reptile, Meta-SGD)

**Risk 3:** Uncertainty adds too much overhead

**Mitigation:**
- Use dropout approximation (faster)
- Focus uncertainty on densification guidance, not final rendering

**Risk 4:** VRAM limitations

**Mitigation:**
- Already solved! Use our RTX 5080 optimizations
- CPU data device, adaptive intervals, expandable segments

### Timeline Risks

**Risk:** Implementation takes longer than expected

**Mitigation:**
- Modular design - each component stands alone
- Can submit LearnedDensify without Uncertainty if needed

**Risk:** Compute experiments take too long

**Mitigation:**
- Use progressive training to speed up all experiments
- Parallel training on multiple scenes

**Risk:** CVPR deadline sooner than expected

**Mitigation:**
- Workshop backup plan (CVPR workshops, ICCV 2025)

---

## Success Criteria

### For CVPR Acceptance (Need at least ONE)
1. ✅ Quantitative: +0.5 dB PSNR over Taming 3DGS
2. ✅ Efficiency: 2× speedup or memory reduction
3. ✅ Novel capability: Uncertainty estimation (first for 3DGS)
4. ✅ Strong ablations: Learned beats all fixed weightings

### Target Metrics

**LearnedDensify:**
- PSNR: **26.5+ dB** (vs 25.93 baseline)
- SSIM: **0.780+** (vs 0.762 baseline)
- LPIPS: **<0.210** (vs 0.223 baseline)
- Convergence: Same quality in **20k vs 30k** iterations

**UncertaintyGS:**
- Uncertainty-error correlation: **Pearson r > 0.7**
- Calibration error: **ECE < 0.05**
- Active learning: **30% reduction** in required views

---

## Next Steps: Immediate Actions

### This Week (Week 1)

**Day 1-2:**
```bash
cd "/home/vortex/Computer Vision/3DGS research/taming-3dgs"
git checkout -b feature/learned-importance
```

1. Create `scene/importance_network.py`:
   ```python
   class ImportanceNet(nn.Module):
       def __init__(self, input_dim=13, hidden_dim=64):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
               nn.Linear(hidden_dim, 32), nn.ReLU(),
               nn.Linear(32, 1), nn.Softplus()
           )

       def forward(self, features):
           return self.net(features)
   ```

2. Modify `utils/taming_utils.py`:
   - Add `compute_gaussian_score_learned()` function
   - Extract features: `[grad, opac, depth, radii, scale, dist, loss, blend, count]`
   - Pass through ImportanceNet
   - Return learned scores

3. Update `train.py`:
   - Initialize ImportanceNet
   - Add to optimizer
   - Call `compute_gaussian_score_learned()` during densification

**Day 3-5:**
- Run sanity check on bicycle dataset
- Visualize learned importance scores
- Compare distributions with hand-crafted scores

**Go/No-Go Decision (End of Week 3):**
- ✅ If learned matches or beats hand-crafted on ANY scene → **Full speed ahead**
- ❌ If completely fails → **Pivot to AdaptiveGS or CompressedGS**

---

## Paper Outline (8 pages)

### Title
"Learning to Densify: Neural Importance Networks for Uncertainty-Aware 3D Gaussian Splatting"

### Abstract (200 words)
- Problem: Hand-crafted importance scoring in 3DGS
- Solution: Learned neural importance network
- Results: +1.2 dB PSNR, generalizes across scenes
- Optional: Uncertainty extension for active learning

### 1. Introduction (1 page)
- 3DGS background and success
- Problem with hand-tuned coefficients (12 parameters, scene-dependent)
- Our contribution: First learned densification
- Key results preview

### 2. Related Work (1 page)
- 3D Gaussian Splatting (Kerbl et al., Taming 3DGS)
- Learned optimization (L2O, VeLO, meta-learning)
- Uncertainty in 3D (NeRF-W, Mega-NeRF)
- Active view selection

### 3. Method (3 pages)
#### 3.1 Background: Taming 3DGS
- Brief review of importance-based densification
- Hand-crafted scoring function

#### 3.2 Neural Importance Networks
- Architecture (Figure 2)
- Feature extraction
- Training strategy

#### 3.3 Meta-Learning for Generalization
- MAML/Reptile overview
- Gradient flow through sampling (Gumbel-Softmax)

#### 3.4 Uncertainty Extension (Optional)
- Probabilistic Gaussians
- Calibration losses
- Active view selection

### 4. Experiments (3 pages)
#### 4.1 Setup
- Datasets (MipNeRF360, Tanks&Temples, DeepBlending)
- Baselines (Taming 3DGS hand-crafted, fixed random weights)
- Metrics (PSNR, SSIM, LPIPS)

#### 4.2 Main Results (Table 1, Figure 3-4)
- Quantitative comparison
- Qualitative visualization
- Convergence curves

#### 4.3 Ablation Studies (Table 2-4)
- Network architecture (32D vs 64D vs 128D)
- Feature importance (which features matter?)
- Meta-learning vs single-scene
- With/without uncertainty

#### 4.4 Generalization
- Zero-shot on new scene types
- Compared to per-scene optimization

### 5. Conclusion (0.5 page)
- Summary of contributions
- Limitations (computational overhead, requires meta-training)
- Future work (extend to dynamic scenes, multi-scale)

### Supplementary Material
- Additional ablations
- Per-scene breakdown
- Video results
- Code release

---

## Comparison Matrix

| Direction | Novelty | Impact | Difficulty | Timeline | CVPR Fit | Recommendation |
|-----------|---------|--------|------------|----------|----------|----------------|
| 1. AdaptiveGS | Medium | High | Low | 10w | Good | **Backup** |
| 2. LearnedDensify | **Very High** | **Very High** | High | 12w | **Excellent** | **PRIMARY** ✅ |
| 3. ProgressiveGS | Medium | Medium | Low | 8w | Good | Helper (speed up experiments) |
| 4. UncertaintyGS | High | High | Medium | 10w | **Excellent** | **SECONDARY** ✅ |
| 5. CompressedGS | Medium | High | Medium | 10w | Good | Workshop/Backup |
| 6. DynamicGS | High | **Very High** | **High** | 12w | **Excellent** | Future work |

---

## Timeline to CVPR 2026

```
October 26, 2025 (Now)
├─ Week 1-3: Implement basic LearnedDensify
├─ Week 4-6: Meta-learning integration
├─ Week 7-9: Uncertainty extension
├─ Week 10-12: Full experiments
├─ Week 13-16: Paper writing
├─ Week 17-20: Iteration + rebuttal prep
└─ November 15, 2025: CVPR 2026 Submission ✅
    ↓
February 2026: Notification
    ↓
June 2026: CVPR Conference
```

**Total Available Time:** ~6 months ✅ **Feasible!**

---

## Key Takeaways

### ✅ We are well-positioned
- Strong baseline implementation (Taming 3DGS on RTX 5080)
- Comprehensive understanding of codebase
- All datasets ready
- Hardware available

### ✅ LearnedDensify is the strongest direction
- Highest novelty (first learned importance for 3DGS)
- Clear motivation (hand-crafted is suboptimal)
- Natural baseline (Taming 3DGS hand-tuned coefficients)
- Aligns with CVPR trends (learned optimization)

### ✅ Timeline is tight but feasible
- 6 months to deadline
- Modular implementation reduces risk
- Multiple backup plans

### 🎯 Recommended Action: **Start Week 1 implementation NOW**

Create branch, implement ImportanceNet, test on bicycle scene. If promising by Week 3, full speed ahead to CVPR 2026!

---

## References & Resources

**Papers:**
- 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)
- Taming 3DGS (Mallick et al., SIGGRAPH Asia 2024)
- MAML (Finn et al., ICML 2017)
- Instant-NGP (Müller et al., SIGGRAPH 2022)

**Code:**
- Taming 3DGS: https://github.com/humansensinglab/taming-3dgs
- learn2learn: https://github.com/learnables/learn2learn
- higher: https://github.com/facebookresearch/higher

**Datasets:**
- MipNeRF360: http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
- Tanks&Temples: https://www.tanksandtemples.org/
- D-NeRF (for DynamicGS): https://github.com/albertpumarola/D-NeRF

---

**Document Created:** October 26, 2025
**For:** CVPR 2026 Submission
**Status:** Ready to implement
**Contact:** Research Team

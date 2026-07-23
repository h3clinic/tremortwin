# References (verified) — synthetic-data CV for hand-tremor frequency & severity

All entries below were verified live (June 2026) by fetching the arXiv/DOI/PMC/GitHub
landing page. Items that could not be verified were dropped.

## A. Video-based & markerless tremor quantification (closest prior art)

1. Pintea S.L., Zheng J., Li X., Bank P.J.M., van Hilten J.J., van Gemert J.C. (2018). **Hand-tremor frequency estimation in videos.** ECCV 2018 Workshop (ECCVW); arXiv:1809.03218. — Canonical RGB-video tremor-frequency method (Lagrangian vs Eulerian) on the public TIM-Tremor set (55 patients); scores frequency "correct" when MAE vs accelerometer < **1 Hz** (the threshold we adopt). https://arxiv.org/abs/1809.03218
2. Williams S., Fang H., Relton S.D., Wong D.C., Alam T., Alty J.E. (2021). **Accuracy of Smartphone Video for Contactless Measurement of Hand Tremor Frequency.** Mov Disord Clin Pract 8(1):69–75. — Optical-flow vs accelerometry; frequency **MAE 0.10 Hz**, <0.5 Hz error in 97% of videos. DOI:10.1002/mdc3.13119. https://pmc.ncbi.nlm.nih.gov/articles/PMC8607978/
3. Williams S., Relton S.D., Fang H., Alty J., Qahwaji R., Graham C.D., Wong D.C. (2020). **Supervised classification of bradykinesia in PD from smartphone videos.** Artif Intell Med 110:101966. — Optical-flow/landmark CV features classify severity; SVM ~0.80 accuracy. DOI:10.1016/j.artmed.2020.101966.
4. Güney G., Jansen T.S., Dill S., Schulz J.B., Dafotakis M., Hoog Antink C., Braczynski A.K. (2022). **Video-Based Hand Movement Analysis of Parkinson Patients … Using High-Frame-Rate Videos and MediaPipe.** Sensors 22(20):7992. — MediaPipe landmarks; tremor-frequency **MAE 0.229 ± 0.174 Hz** vs accelerometer. DOI:10.3390/s22207992. https://pmc.ncbi.nlm.nih.gov/articles/PMC9611677/
5. Park K.W. et al. (2021). **Machine Learning-Based Automatic Rating for Cardinal Symptoms of PD.** Neurology 96(13):e1761–e1769. — OpenPose kinematics + SVM auto-rate rest tremor; agreement κ=0.791, ICC=0.927. DOI:10.1212/WNL.0000000000011654.
6. Liu W. et al. (2023). **Vision-based estimation of MDS-UPDRS scores for quantifying PD tremor severity.** Med Image Anal 85:102754. — EVM + temporal-shift net predicts MDS-UPDRS tremor; **90.6% accuracy** hand rest tremor, 84.9% postural. DOI:10.1016/j.media.2023.102754.
7. **Friedrich M.U. et al. (2024). Validation and application of computer vision algorithms for video-based tremor analysis.** npj Digit Med 7:165. — *The closest published instance of this exact pipeline* (MediaPipe landmarks → frequency + amplitude), validated vs accelerometry; postural-amplitude ρ=0.86, frequency MAE ≈0.2 Hz. DOI:10.1038/s41746-024-01153-1. https://pmc.ncbi.nlm.nih.gov/articles/PMC11192937/
8. Duque-Quiceno F., Sarapata G., Dushin Y., Allen M., O'Keeffe J. (2024). **Deep learning for objective estimation of Parkinsonian tremor severity.** arXiv:2409.02011. — Pixel-based DL postural-tremor scoring, 2,742 assessments / 5 centers. https://arxiv.org/abs/2409.02011
9. Zhang L., Yadav V., Koesmahargyo V., Abbas A., Galatzer-Levy I. (2021). **Prediction of clinical tremor severity using Rank Consistent Ordinal Regression.** arXiv:2105.01133. — Ordinal DL on 276 ET videos; **TETRAS MAE 0.45**, most errors between adjacent labels (motivates our ±1-bin metric). https://arxiv.org/abs/2105.01133
10. Netukova S. et al. (2026). **A Vision-Based Algorithm for Assessing Head and Hand Tremor: … Validation Against IMU Sensors.** Sensors 26(3):928. — 2D-video center-of-mass + spectral vs IMU; peak-frequency ICC 0.60–0.67 (hands). DOI:10.3390/s26030928.
11. **Wolke R., Welzel J., Maetzler W., Deuschl G., Becktepe J.** (2025). **Validity of tremor analysis using smartphone compatible computer vision frameworks.** Sci Rep 15(1):13391. — Benchmarks MediaPipe & Apple Vision vs motion capture/accelerometry/TETRAS in 20 patients; peak frequency accurate but **amplitude accuracy vs OMC insufficient for clinical use** (our headline sim-to-real caveat). Renders a rigged 3D hand in **Blender** to *validate/calibrate the detectors* (validation-only — **no model is trained on the synthetic hands**), which is precisely the boundary our train-on-synthetic contribution crosses. DOI:10.1038/s41598-025-97252-4. https://www.nature.com/articles/s41598-025-97252-4  *(Note: earlier drafts mis-attributed this to "Güney 2025"; Güney et al. 2022 [#4] is a different, real paper.)*
12. Guarin D.L. et al. (2025). **Video-based quantification of hand postural tremor without external references (visionMD).** npj Parkinsons Dis 11:351. DOI:10.1038/s41531-025-01196-5.
13. Acevedo G.T. et al. (2025). **VisionMD: an open-source tool for video-based analysis of motor function in movement disorders.** npj Parkinsons Dis 11. DOI:10.1038/s41531-025-00876-6.
14. Lee S.H. et al. (2024). **Quantification of tremor dynamics via video-based analysis.** Multimed Tools Appl. DOI:10.1007/s11042-024-18438-y.

## B. Clinical tremor scales & physiology (Methods grounding)

1. Elble R. et al. (2012). **Reliability of a new scale for essential tremor (TETRAS).** Mov Disord 27(12):1567–1569. DOI:10.1002/mds.25162.
2. Goetz C.G. et al. (2008). **MDS-UPDRS: Scale presentation and clinimetric testing.** Mov Disord 23(15):2129–2170. DOI:10.1002/mds.22340. — Rest (3.17)/postural (3.15)/kinetic (3.16) hand-tremor items, 0–4.
3. Fahn S., Tolosa E., Marin C. (1993). **Clinical rating scale for tremor (Fahn–Tolosa–Marin / CRST).** In *Parkinson's Disease and Movement Disorders*, 2nd ed., 271–280.
4. Bain P.G., Findley L.J. et al. (1993). **Assessing tremor severity.** J Neurol Neurosurg Psychiatry 56(8):868–873. DOI:10.1136/jnnp.56.8.868.
5. Elble R., Bain P., Forjaz M.J. et al. (2013). **Task force report: Scales for screening and evaluating tremor.** Mov Disord 28(13):1793–1800. DOI:10.1002/mds.25648.
6. Bhatia K.P., Bain P., Bajaj N., Elble R.J., Hallett M., Louis E.D., Raethjen J., Stamelou M., Testa C.M., Deuschl G.; IPMDS Tremor Task Force (2018). **Consensus Statement on the Classification of Tremors.** Mov Disord 33(1):75–87. DOI:10.1002/mds.27121. — Current tremor taxonomy.
7. Deuschl G., Bain P., Brin M. (1998). **Consensus statement of the MDS on Tremor.** Mov Disord 13(S3):2–23. DOI:10.1002/mds.870131303.
8. Lenka A., Jankovic J. (2021). **Tremor Syndromes: An Updated Review.** Front Neurol 12:684835. — Frequency bands: essential/action ~4–12 Hz, physiologic ~8–12 Hz, Parkinsonian rest ~4–6 Hz (our decoupled-generator bands). DOI:10.3389/fneur.2021.684835.
9. Zhang J., Xing Y., Ma X., Feng L. (2017). **Differential Diagnosis of PD, ET, and Enhanced Physiological Tremor with EMG.** Parkinsons Dis 2017:1597907. DOI:10.1155/2017/1597907.

## C. Synthetic data / domain randomization / sim-to-real

1. Tobin J. et al. (2017). **Domain Randomization for Transferring DNNs from Simulation to the Real World.** IROS 2017; arXiv:1703.06907. — Foundational DR.
2. Tremblay J. et al. (2018). **Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization.** CVPRW 2018; arXiv:1804.06516.
3. Prakash A. et al. (2019). **Structured Domain Randomization.** ICRA 2019; arXiv:1810.10093.
4. OpenAI; Andrychowicz M. et al. (2020). **Learning Dexterous In-Hand Manipulation.** IJRR 39(1):3–20; arXiv:1808.00177. — DR of physics + vision on a robot hand, zero-shot to reality.
5. Zhao W., Queralta J.P., Westerlund T. (2020). **Sim-to-Real Transfer in Deep RL for Robotics: a Survey.** arXiv:2009.13303.
6. Varol G. et al. (2017). **Learning from Synthetic Humans (SURREAL).** CVPR 2017; arXiv:1701.01370.
7. Zimmermann C., Brox T. (2017). **Learning to Estimate 3D Hand Pose from Single RGB Images (RHD).** ICCV 2017; arXiv:1705.01389.
8. Mueller F. et al. (2018). **GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB.** CVPR 2018; arXiv:1712.01057. — Photorealism transfer to close the sim-to-real gap for RGB hands.
9. Zimmermann C. et al. (2019). **FreiHAND.** ICCV 2019; arXiv:1909.04349.
10. Marquez Chavez J., Tang W. (2022). **A Vision-Based System for Stage Classification of Parkinsonian Gait Using ML and Synthetic Data.** Sensors 22(12):4463. — Precedent for synthetic-data-trained movement-disorder CV. DOI:10.3390/s22124463.

## D. Useful GitHub repositories (verified)

**Landmark tracking**
- google-ai-edge/mediapipe (~35.9k★, C++) — hand-landmarker (21 landmarks) used by the render-side pipeline. https://github.com/google-ai-edge/mediapipe
- CMU-Perceptual-Computing-Lab/openpose (~34.2k★) — alt hand keypoints for cross-validation.
- open-mmlab/mmpose (~7.7k★) — trainable 2D hand-keypoint backbones (FreiHand, InterHand2.6M).
- DeepLabCut/DeepLabCut (~5.7k★) — markerless custom-landmark tracking.
- gianluca-amprimo/GMH-D (~17★) — depth-enhanced MediaPipe Hands, validated for PD hand assessment. https://github.com/gianluca-amprimo/GMH-D

**Tremor analysis from signals/video**
- tesar-tech/TremAn3 — video tremor-frequency via center-of-motion. https://github.com/tesar-tech/TremAn3
- jamesbungay/cv-tremor-amplitude — MediaPipe+OpenCV PD tremor amplitude from webcam video. https://github.com/jamesbungay/cv-tremor-amplitude
- cokelaer/spectrum (~373★) — Welch/parametric PSD for dominant-frequency estimation. https://github.com/cokelaer/spectrum
- sjmercer65/tremor — accelerometer tremor features / PD modelling.

**Synthetic hands / rigged models / Blender rendering**
- otaheri/MANO (~419★), hassony2/manopth (~690★), lixiny/manotorch (~288★) — parametric/differentiable rigged hand meshes for controlled joint-angle tremor injection.
- hassony2/obman_render (~80★) — Blender renderer: MANO → synthetic hand images + GT keypoints with texture/lighting randomization. https://github.com/hassony2/obman_render
- kaiidams/FreeHand-Dataset — Blender scripts rendering annotated synthetic hands.
- facebookresearch/InterHand2.6M (~753★), adwardlee/RenderIH (~285★) — realistic hand-pose distributions / domain-randomized renderer.

**Accelerometer/IMU tremor (gold-standard reference)**
- funsaized/PD-and-ET-Tremor-Quantification — amplitude + Fourier frequency with MDS-UPDRS integration.
- NikhilMahadevan/analyze-tremor-bradykinesia-PD (~27★) — wrist-accelerometer resting-tremor analytics.
- biomarkersParkinson/paradigma (~17★) — maintained digital tremor/gait biomarker toolbox.

## E. Novelty positioning / closest prior art (from targeted novelty search)

- **Wolke R. et al. (2025)** — see A#11. Closest prior work: renders a Blender hand to *validate* CV detectors, **does not train on it**. The train-vs-validate boundary we cross.
- **Pan M-K. (2025). Targeting the fundamentals for tremors: the frequency and amplitude coding in essential tremor.** J Biomed Sci 32:18. DOI:10.1186/s12929-024-01112-8. — Tremor **frequency and amplitude arise from distinct, independent neural mechanisms** → physiological justification for modeling them as independent factors (our decoupled generator) and against the repo's single-scalar coupling.
- **(bioRxiv 2025) Muscle activation prediction in essential tremor through neuromusculoskeletal digital twinning and deep neural networks.** DOI:10.64898/2025.12.17.693373. — The only work training *purely on a tremor digital twin*, but predicts **muscle activation from wrist-angle** (not freq/severity, not CV, no sim-to-real). Adjacent, non-overlapping.
- **"Conditional TremorGANs" (2025). A GAN-Based Method for Synthesizing Tremor Data in Parkinson's Disease.** Cognitive Computation 17. DOI:10.1007/s12559-025-10533-y. — Synthetic **accelerometer** tremor to *augment* real data for **classification** (not CV, not regression). *(Paywalled — abstract only.)*
- **De A. et al. (2023). Machine Learning in Tremor Analysis: Critique and Directions.** Movement Disorders 38(5). DOI:10.1002/mds.29376. — Catalogs ML-tremor pitfalls; does **not** raise the frequency↔severity target-collinearity leakage (our finding). *(Paywalled — abstract only; confirm before citing as "does not mention".)*
- Anicet Zanini & Colombini (2020). **Parkinson's Disease EMG Data Augmentation … with DCGANs and Style Transfer.** Sensors 20(9):2605. — synthetic **EMG** tremor augmentation; not CV. https://pmc.ncbi.nlm.nih.gov/articles/PMC7248755/

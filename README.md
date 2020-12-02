# SCALPELS
Self-Correlation Analysis of Line Profiles for Extracting Low-amplitude Shifts

  
An implementation of the algorithm proposed by [Collier Cameron et al. 2020](https://arxiv.org/abs/2011.00018).
<a href="https://commons.wikimedia.org/wiki/File:Dessin_scalpel.svg"> <img align="right" width="180" height="180" alt="Dessin scalpel" title="Petit B, CC BY-SA 4.0, via Wikimedia Commons" src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Dessin_scalpel.svg/256px-Dessin_scalpel.svg.png"></a>


- [x] Calculate ACFs from CCFs
- [x] SVD decomposition of ACFs
- [x] Project observed RVs onto k principal components
- [x] Estimate optimal k using BIC
- [x] Model stellar variability and planetary motion simultaneously
  - [ ] Calculate covariance matrix following appendix B


# InfPolyn
Source code and experiments for the paper, 
InfPolyn, a Nonparametric Bayesian Characterization for Composition-Dependent Interdiffusion Coefficients (https://www.mdpi.com/1169732).

This codes aim to provide a statistical framework to characterize the component-dependent interdiffusion coefficients in the Boltzmann–Matano analysis. 

From the perspective of machine learning model, InfPolyn is a mixture of Gaussian processes (GPs), each of which is multiplied by a known derivative function and is equipped with a proper Laplace prior.


---
### Repository structure
```
model: interdiffusion forward solver and fitting process 
ternary: experimental code and data for the ternary experiments in the paper
quaternary: experimental code and data for the quaternary experiments in the paper
utils: sources codes and other Boltzmann–Matano analysis codes
```

- model: interdiffusion forward solver and fitting process 
- ternary: experimental code and data for the ternary experiments in the paper
- quaternary: experimental code and data for the quaternary experiments in the paper
- utils: sources codes and other Boltzmann–Matano analysis codes

---
### Citation
Xing, Wei W., et al. "InfPolyn, a Nonparametric Bayesian Characterization for Composition-Dependent Interdiffusion Coefficients." Materials 14.13 (2021): 3635. 
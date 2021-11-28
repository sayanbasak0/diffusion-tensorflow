# Diffusion Simulator/Animator with NumPy

### Current Status:
* NumPy version :
    * Simulation works for N-D diffusion.
    * Animation works in 1-D and 2-D diffusion.
    * _Checkpointing_ functionality added.
    * _Experimental_: 3-D diffusion animation.
* TensorFlow-v1:
    * Simulation works for N-D diffusion.
    * Animation for 2-D diffusion in `tensorflow-v1/diffusion_tenorflow1.ipynb`.

---

## Diffusion equation:
### The continuity equation:
$$\nabla \cdot \vec{j} + \frac{\partial \phi}{\partial t} = 0$$

$\phi$ : Density of diffusing quantity

$\vec{j}$ : Current density

### Fick's first law:
$$\vec{j} = - \overleftrightarrow{D}[\phi,\vec{r}] \nabla\phi[\vec{r},t]$$
$\overleftrightarrow{D}[\phi,\vec{r}]$ : Diffusion (tensor) coefficient for density $\phi$ at position $\vec{r}$

Combining the above equations gives the general diffusion equation:
$$\frac{\partial\phi}{\partial t} = \nabla \Big( \overleftrightarrow{D}[\phi,\vec{r}] \nabla\phi \Big) = \partial^{\alpha} \big(D_{\alpha\beta}\partial^{\beta} \phi \big)$$ 

### The typical diffusion:
When D is an scalar constant(isotropic and independent of density and position), the diffusion equation reduces to:
$$\frac{\partial\phi}{\partial t} = D \nabla^2 \phi$$ 

## This Implementation:
$$\overleftrightarrow{D}[\phi,\vec{r}] \equiv D[\vec{r}]\bigg(1-\frac{B[\vec{r}]}{|\nabla \phi|}\bigg)\Theta[|\nabla\phi|-B[\vec{r}]] $$ 
Here, the diffusion coefficient is isotropic, hence is a scalar quatity
The simulation here evolves the diffusing quantity given:
1. The boundary conditions
2. The initial distribution
3. The diffusion coefficient, $D[\vec{r}]$, as function of position
4. Minimum gradient, $B[\vec{r}]$, required to drive current


# Linear Plasma Distribution Functions 

$$
\frac{d f_1}{d t}=-\frac{q}{m}(\vec E_1 + \vec v \times \vec B_1)\cdot \frac{\partial f_0}{\partial t}\\
$$

Introducing $f_1(v_\perp, v_\parallel, \phi)$ where $\phi$ is the gyro angle
$$
i(\omega - \vec v \cdot \vec k) f_1 + \Omega \frac{\partial f_1}{\partial \phi}=\frac{q}{m}\left(\left(1-\frac{\vec v \cdot \vec k}{\omega}\right)\vec E_1 + \frac{\vec v\cdot \vec E_1}{\omega}\vec k\right)\cdot \nabla_v f_0\\
$$
Also
$$
f_1(v_\bot, v_\parallel, \phi) = \sum_{n=-\infty}^\infty g_n(v_\bot, v_\parallel)\exp(-i n\phi)
$$

$$
\vec E\cdot \vec v = E_x v_\bot \cos \phi + E_y v_\bot \sin \phi + E_z v_\parallel\\ \vec k\cdot \vec v = k_\bot v_\bot \cos \phi + k_\parallel v_\parallel
$$

So
$$
(\omega - \vec v \cdot \vec k - i \Omega\frac{\partial}{\partial \phi}) f_1 =-i\frac{q}{m\omega}\left(\left(\omega-\vec v \cdot \vec k\right)\vec E_1 + \left(\vec v\cdot \vec E_1\right)\vec k \right)\cdot \nabla_v f_0\\
$$
We also know
$$
\vec{k}\cdot\vec{\nabla_{v}}=k_{\parallel}\frac{\partial}{\partial v_{z}}+k_{\perp}\frac{\partial}{\partial v_{x}}=k_{\parallel}\frac{\partial}{\partial v_{z}}+k_{\perp}\cos\phi\frac{\partial}{\partial v_{\perp}}-\frac{k_{\perp}\sin\phi}{v_{\perp}}\frac{\partial}{\partial\phi}\\\vec{E}\cdot\vec{\nabla_{v}}=E_{z}\frac{\partial}{\partial v_{z}}+E_{x}\cos\phi\frac{\partial}{\partial v_{\perp}}-\frac{E_{x}\sin\phi}{v_{\perp}}\frac{\partial}{\partial\phi}+E_{y}\sin\phi\frac{\partial}{\partial v_{\perp}}+E_{y}\frac{\cos\phi}{v_{\perp}}\frac{\partial}{\partial\phi}
$$
hence
$$
(\omega - \vec v \cdot \vec k - i \Omega\frac{\partial}{\partial \phi}) f_1 =-i\frac{q}{m\omega}\left[\left(\omega-\vec v \cdot \vec k\right)\vec E_1\cdot \nabla_v + (\vec v\cdot \vec E_1)\vec k\cdot \nabla_v \right] f_0\\
$$
 and introducing $\partial f_0/\partial \phi = 0$, expanding and simplifying 
$$
\left[\left(\omega-\vec v \cdot \vec k\right)\vec E_1\cdot \nabla_v \right] + \left[\vec v\cdot \vec E_1\vec k\cdot \nabla_v \right] =\\\left(\omega-k_\bot v_\bot \cos \phi - k_\parallel v_\parallel\right)\left(E_{z}\frac{\partial}{\partial v_{z}}+(E_{x}\cos\phi+E_{y}\sin\phi)\frac{\partial}{\partial v_{\perp}}\right) + \\(E_x v_\bot \cos \phi + E_y v_\bot \sin \phi + E_z v_\parallel)\left(k_{\parallel}\frac{\partial}{\partial v_{z}}+k_{\perp}\cos\phi\frac{\partial}{\partial v_{\perp}}\right)
$$
Turning to sympy

```python
import sympy
from sympy import sin, cos
kz, kr, vz, vr, Ex, Ey, Ez, w, p, dr, dz = sympy.var('kz kr vz vr Ex Ey Ez w p dr dz')
expr = (w - kr * vr * cos(p) - kz * vz) * ((Ex * cos(p) + Ey * sin(p))*dr + Ez * dz) + (Ex * vr * cos(p) + Ey * vr * sin(p) + Ez * vz) * (kz * dz + kr * cos(p) * dr)
```

$$
(k_z v_\bot \cos\phi E_x + k_z v_\bot \sin\phi E_y + (\omega - k_\bot v_\bot) E_z)\frac{\partial}{\partial v_\parallel}+\\((\omega - k_\parallel v_z) \cos\phi E_x + (\omega - k_z v_z)\sin\phi E_y + k_\bot v_z\cos\phi E_z)\frac{\partial}{\partial v_\bot}
$$

Let
$$
\vec L_\parallel = (k_z v_\bot \cos\phi \hat e_x + k_z v_\bot \sin\phi \hat e_y + (\omega - k_\bot v_\bot) \hat e_z)\\\vec L_\bot = ((\omega - k_\parallel v_z) \cos\phi \hat e_x + (\omega - k_z v_z)\sin\phi \hat e_y + k_\bot v_z\cos\phi \hat e_z)
$$
So
$$
(\omega - \vec v \cdot \vec k - i \Omega\frac{\partial}{\partial \phi}) f_1 = \sum_n(\omega - \vec v \cdot \vec k - \Omega n) g_n \exp(-in\phi) =-i\frac{q}{m\omega}\left[\vec L_\parallel \cdot \vec E_1 \frac{\partial}{\partial v_\parallel} + \vec L_\bot \cdot \vec E_1 \frac{\partial}{\partial v_\bot}\right] f_0\\
$$
Which separates out the different perturbations in gyroangle
$$
g_n \propto -i\frac{q \exp(in\phi)}{m\omega (\omega - k_\bot v_\bot \cos \phi - k_\parallel v_\parallel - n\Omega)}\sum_{i={\parallel,\bot}}\vec L_i \cdot \vec E_1 \frac{\partial f_0}{\partial v_i}\\
$$
Given that it's linear $\frac{q}{m}$ can be replaced by $\Omega$ because the amplitude of $f_1$ is arbitrary as long as its $\ll f_0$,
$$
g_n \propto -i\frac{\Omega \exp(in\phi)}{\omega (\omega - k_\bot v_\bot \cos \phi - k_\parallel v_\parallel - n\Omega)}\sum_{i={\parallel,\bot}}\vec L_i \cdot \vec E_1 \frac{\partial f_0}{\partial v_i}.\\
$$


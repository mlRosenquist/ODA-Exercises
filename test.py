# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Lec 3  
# %% [markdown]
# ## 5 CZ 
# 
# ### 5.9
# ![image info](./images/CZ_5_9.png)
# 
# $$x^2_1-x^2_2=12$$ 
# $$x_2=8/x_1$$
# $$x^2_1-8/x_1=12$$
# $$x^4_1-12x^2_1-64=0$$
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu = 0
std = 1

p1d = np.poly1d([1, 0, -12, -64])
p = np.polynomial.Polynomial(p1d.coef[::-1])

x = np.linspace(start=-4, stop=4, num=100)
y = stats.norm.pdf(x, mu, std) 
plt.plot(x_1, x_2)
plt.show()

# %% [markdown]
# ### 5.10 a
# ![image info](./images/CZ_5_10a.png)
# 
# ### 5.10 b
# ![image info](./images/CZ_5_10b.png)
# 
# ## 6 CZ
# 
# ### 6.3 
# ![image info](./images/CZ_6_3.png)
# 
# ### 6.8 
# ![image info](./images/CZ_6_8.png)
# 
# ### 6.10
# ![image info](./images/CZ_6_10.png)
# 
# ### 6.11
# ![image info](./images/CZ_6_11.png)
# 
# ### 6.20
# ![image info](./images/CZ_6_20.png)
# 
# ### 6.23 
# ![image info](./images/CZ_6_23.png)
# 
# ## Quadratic Forms
# ![image info](./images/QF.png)
# ### 10
# 
# ### 11 
# 
# ### 12


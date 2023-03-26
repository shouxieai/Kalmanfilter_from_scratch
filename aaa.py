# %%
from collections import namedtuple

gaussian = namedtuple("Gaussian", ['mean', 'var'])
gaussian.__repr__ = lambda s: f"N(mean = {s[0]:.3f}, var={s[1]:.3f})"


# %%
g1 = gaussian(3.4, 10.1)
g2 = gaussian(mean = 4.5, var = 0.2**2)

print(g1)
print(g2)
g1.mean

# %%
g1[0], g1[1]

# %%
def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)

# %%
pos = gaussian(10., 0.2**2)
move = gaussian(15., 0.7 ** 2)
predict(pos, move)

# %%
def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)

def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

predicted_pos = gaussian(10.0, 0.2**2)
measured_pos  = gaussian(11.0, 0.1**2)

estimated_pos = update(predicted_pos, measured_pos)

print(predicted_pos)
print(measured_pos)
print(estimated_pos)


# %%
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt


motion_var = 1.0
sensor_var = 2.0

x          = gaussian(0, 20.**2)
velocity   = 1.0
dt         = 1
motion_model = gaussian(velocity, motion_var)

zs = []
current_x = x.mean
for _ in range(20):
    v = velocity + randn()*motion_var
    current_x += v*dt

    measurement = current_x + randn()*sensor_var
    zs.append(measurement)



# %%
def print_result(predict, update, z, epoch):

    # 细节暂不需要深究

    # predicted_pos, updated_posclear, measured_pos

    predict_template = '{:3.0f} {: 7.3f} {: 8.3f}'

    update_template  = '\t{: .3f}\t{: 7.3f} {: 7.3f}'

    print(predict_template.format(epoch, predict[0], predict[1]),end='\t')

    print(update_template.format(z, update[0], update[1]))


def plot_result(epochs ,prior_list, x_list, z_list):

    epoch_list = np.arange(epochs)

    plt.plot(epoch_list, prior_list, linestyle=':', color='r',label = "prior/predicted_pos", lw=2)

    plt.plot(epoch_list, x_list, linestyle='-', color='g', label = "posterior/updated_pos",lw=2)

    plt.plot(epoch_list, z_list, linestyle=':', color='b', label = "likelihood/measurement", lw=2)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
prior_list, x_list, z_list = [], [], []

print("epoch\tPREDICT\t\t\tUPDATE")
print('     \tx      var\t\t  z\t    x      var')

for epoch, z in enumerate(zs):
    prior = predict(x, motion_model)
    likelihood = gaussian(z, sensor_var)

    x = update(likelihood, prior)

    print_result(prior, x, z, epoch)
    prior_list.append(prior.mean)
    x_list.append(x.mean)
    z_list.append(z)



print(f"final estimate:       {x.mean:10.3f}")


# plot_result(10, prior_list, x_list, z_list)


# %%




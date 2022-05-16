import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises



# phi = np.linspace(0,1, 1000)
# r=.6; cycle_offset=.5
# a=0; b=a+r; k=100

# ca_right = -vonmises.pdf(phi, k, loc=a, scale=1/(2*np.pi))
# cb_right = vonmises.pdf(phi, k, loc=b, scale=1/(2*np.pi))
# c_sum_right = np.cumsum(ca_right+cb_right)
# c_scale_right = -(c_sum_right.max()-c_sum_right.min())/2
# c_final_right = ((c_sum_right/c_scale_right))/2
# c_final_right = c_final_right-c_final_right.min()

# ca_left = -vonmises.pdf(phi+cycle_offset, k, loc=a, scale=1/(2*np.pi))
# cb_left = vonmises.pdf(phi+cycle_offset, k, loc=b, scale=1/(2*np.pi))
# c_sum_left = np.cumsum(ca_left+cb_left)
# c_scale_left = -(c_sum_left.max()-c_sum_left.min())/2
# c_final_left = ((c_sum_left/c_scale_left))/2
# c_final_left = c_final_left-c_final_left.min()



# plt.figure(2)
# plt.plot(phi, c_final_right)
# plt.plot(phi, c_final_left)
# plt.grid()
# plt.show()



xall=np.linspace(0,10,10000)
y=[]
for x in xall: 
    y.append(int((x%1)*(1000)))

plt.plot(x)
plt.plot(y)
plt.show()
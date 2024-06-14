from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)
(1797, 64)
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[0])

plt.show()
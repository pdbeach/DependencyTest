import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Known values
TP = 192
FN = 0
FP = 1

# If you do not know TN, you can set it to 0 or any placeholder:
TN = 0

# Build confusion matrix
cm = np.array([[TP, FN],
               [FP, TN]])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['Actual Positive',   'Actual Negative'])
plt.title('Confusion Matrix (Partial Data)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
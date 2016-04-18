import numpy as np
import matplotlib.pylab as plt
import json



with open('train_process.json', 'r') as f:
    train_process = json.load(f)
f.close()


with open('train_process_with_alpha.json', 'r') as f:
    train_process_alpha = json.load(f)
f.close()


fig, ax2 = plt.subplots(1, 1)
it_counts = train_process['it_counts']

# ax1.plot(it_counts[1:], train_process['loss_values'][1:], label = 'Loss')
# ax1.plot(it_counts[1:], train_process_alpha['loss_values'][1:], label = 'Loss with l2 regularization')
# ax1.legend(loc = 'best', fontsize = 'medium')


ax2.plot(it_counts, train_process['train_scores'], label = 'Training accuracy')
ax2.plot(it_counts, train_process['test_scores'], label = 'Test accuracy')
ax2.plot(it_counts, train_process_alpha['train_scores'], label = 'Training accuracy with l2 regularization')
ax2.plot(it_counts, train_process_alpha['test_scores'], label = 'Test accuracy with l2 regularization')
ax2.legend(loc = 'best', fontsize = 'medium')
ax2.grid(True)

plt.show()
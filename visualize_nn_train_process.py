import matplotlib.pylab as plt
import json

fig, ax2 = plt.subplots(1, 1)

for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
    with open('train_process_learning_rate_' + learning_rate + '.json', 'r') as f:
        train_process = json.load(f)
    f.close()

    it_counts = train_process['it_counts']
    
    # ax1.plot(it_counts[1:], train_process['loss_values'][1:], label = 'Loss')
    # ax1.plot(it_counts[1:], train_process_alpha['loss_values'][1:], label = 'Loss with l2 regularization')
    # ax1.legend(loc = 'best', fontsize = 'medium')
    
    ax2.plot(it_counts, train_process['train_scores'], label = 'Training accuracy - learning rate = ' + learning_rate)
    ax2.plot(it_counts, train_process['test_scores'], label = 'Test accuracy - learning rate = ' + learning_rate)

ax2.legend(loc = 'best', fontsize = 'medium')
ax2.grid(True)

plt.show()
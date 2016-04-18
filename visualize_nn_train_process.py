import json
import matplotlib.pylab as plt
import itertools
import seaborn

fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)

palette = itertools.cycle(seaborn.color_palette(n_colors = 10))

for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
    with open('train_process_learning_rate_' + str(learning_rate) + '.json', 'r') as f:
        train_process = json.load(f)
    f.close()

    it_counts = train_process['it_counts'][1:]
    
    # ax1.plot(it_counts, train_process['loss_values'][1:], label = 'Loss--learning rate = ' + str(learning_rate), c = next(palette))
    
    if learning_rate != 0.1:
        ax2.plot(it_counts, train_process['train_scores'][1:], label = 'Training accuracy-learning rate = ' + str(learning_rate), c = next(palette))
        ax2.plot(it_counts, train_process['test_scores'][1:], label = 'Test accuracy-learning rate = ' + str(learning_rate), c = next(palette))
    else:
        ax2.plot(it_counts, train_process['train_scores'][1:], '--', label = 'Training accuracy-learning rate = ' + str(learning_rate), c = next(palette))
        ax2.plot(it_counts, train_process['test_scores'][1:], '--', label = 'Test accuracy-learning rate = ' + str(learning_rate), c = next(palette))

ax1.set_xlabel('Number of steps', fontsize = 'medium')
ax1.legend(loc = 'best', fontsize = 'medium')
ax1.grid(True)

ax2.set_xlabel('Number of steps', fontsize = 'medium')
ax2.legend(loc = 'best', fontsize = 'medium')
ax2.grid(True)

plt.show()
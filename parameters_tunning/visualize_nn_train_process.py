import json
import matplotlib.pylab as plt
import itertools
import seaborn

fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)

palette = itertools.cycle(seaborn.color_palette(n_colors = 10))

for hidden1 in [10, 50, 100, 150]:
    with open('train_process_hidden1_' + str(hidden1) + '.json', 'r') as f:
        train_process = json.load(f)
    f.close()

    it_counts = train_process['it_counts'][1:]
    
    # ax1.plot(it_counts, train_process['loss_values'][1:], label = 'Loss--learning rate = ' + str(learning_rate), c = next(palette))
    
    if hidden1 != 150:
        ax2.plot(it_counts, train_process['train_scores'][1:], label = 'Training accuracy-hidden1 = ' + str(hidden1), c = next(palette))
        ax2.plot(it_counts, train_process['test_scores'][1:], label = 'Test accuracy-hidden1 = ' + str(hidden1), c = next(palette))
    else:
        ax2.plot(it_counts, train_process['train_scores'][1:], '--', label = 'Training accuracy-hidden1 = ' + str(hidden1), c = next(palette))
        ax2.plot(it_counts, train_process['test_scores'][1:], '--', label = 'Test accuracy-hidden1 = ' + str(hidden1), c = next(palette))

ax1.set_xlabel('Number of steps', fontsize = 'medium')
ax1.legend(loc = 'best', fontsize = 'medium')
ax1.grid(True)

ax2.set_xlabel('Number of steps', fontsize = 'medium')
ax2.legend(loc = 'best', fontsize = 'medium')
ax2.grid(True)

plt.show()
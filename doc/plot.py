import matplotlib.pyplot as plt

result = """
    1       97                4.37                 0.2109
    2      194                7.77                 0.7620
    3      291               11.16                 0.8764
    4      388               14.54                 0.8979
    5      485               17.93                 0.9098
    6      582               21.32                 0.9177
    7      679               24.71                 0.9280
    8      776               28.09                 0.9332
    9      873               31.48                 0.9395
   10      970               34.86                 0.9430
"""

rows = []
for row in result.strip().split("\n"):
    numbers = [float(x) for x in row.split()]
    rows.append(numbers)

epoch, batch, t, accuracy = map(list, zip(*rows))

plt.plot(epoch, [100 - 100 * x for x in accuracy])
plt.xticks(epoch)
plt.xlabel("Epoch")
plt.ylabel("Validation error [%]")
plt.savefig("a100_epoch_vs_validation_error.png")
plt.show()

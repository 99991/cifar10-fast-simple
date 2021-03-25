import matplotlib.pyplot as plt

result = """
epoch    batch    total time [sec]    validation accuracy
    1       97                8.50                 0.2109
    2      194               12.09                 0.7620
    3      291               15.75                 0.8764
    4      388               19.41                 0.8979
    5      485               23.07                 0.9098
    6      582               26.74                 0.9177
    7      679               30.40                 0.9280
    8      776               34.06                 0.9332
    9      873               37.72                 0.9395
   10      970               41.38                 0.9430
"""

rows = []
for row in result.split("validation accuracy")[-1].split("\n"):
    row = row.strip()

    if len(row) == 0: continue

    numbers = [float(x) for x in row.split()]
    rows.append(numbers)

epoch, batch, t, accuracy = map(list, zip(*rows))

plt.plot(epoch, [100 - 100 * x for x in accuracy])
plt.xticks(epoch)
plt.xlabel("Epoch")
plt.ylabel("Validation error [%]")
plt.savefig("a100_epoch_vs_validation_error.png")
plt.show()

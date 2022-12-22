import matplotlib.pyplot as plt

baseline, model = [], []
with open("lstm_burst_model_output/out_with_std.csv", "r") as f:
    for line in f:
        s = line.split(",")
        baseline.append((float(s[1]), float(s[2])))
        model.append((float(s[3]), float(s[4])))

labels = list(range(1, 32))

baseline, baseline_err = list(zip(*baseline))
model, model_err = list(zip(*model))

plt.errorbar(
    labels,
    baseline,
    baseline_err,
    fmt="o",
    capsize=3,
    label="Viterbi Decoder (AWGN, D=15)",
    color=(51 / 255, 51 / 255, 178 / 255),
)
plt.errorbar(
    labels,
    model,
    model_err,
    fmt="o",
    capsize=3,
    label="4-Layer Bidirectional LSTM (units=64)",
    color="orange",
)
plt.xlabel("Burst Length")
plt.ylabel("Average Hamming Distance")
plt.legend(loc="upper left")
plt.show()

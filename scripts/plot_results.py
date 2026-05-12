import os
import matplotlib.pyplot as plt

# Results from experiments
batch_sizes = [8, 16, 32]

dense_throughput = [359.07, 702.76, 1274.51]
sparse_throughput = [360.50, 687.70, 1204.58]

dense_ppl = [14.205, 12.734, 14.217]
sparse_ppl = [15.007, 13.505, 15.053]

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(6.5, 4.2))

plt.plot(
    batch_sizes,
    dense_throughput,
    marker="o",
    linewidth=2,
    label="Dense baseline"
)

plt.plot(
    batch_sizes,
    sparse_throughput,
    marker="s",
    linewidth=2,
    label="Shared-mask sparse"
)

plt.xlabel("Batch Size")
plt.ylabel("Throughput (tokens/sec)")
plt.title("Dense vs. Shared-Mask Sparse Decoding Throughput")
plt.xticks(batch_sizes)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("figures/throughput_vs_batch.pdf", bbox_inches="tight")
plt.savefig("figures/throughput_vs_batch.png", dpi=300, bbox_inches="tight")

print("Saved figures/throughput_vs_batch.pdf")
print("Saved figures/throughput_vs_batch.png")
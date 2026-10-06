import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

delivery_times = np.array([
    25, 30, 28, 35, 32, 27, 40, 22, 31, 29,
    36, 34, 26, 38, 33, 24, 30, 29, 35, 31,
    28, 32, 27, 36, 34, 30, 25, 29, 33, 31,
    37, 26, 28, 35, 30, 32, 24, 27, 34, 29,
    31, 36, 28, 33, 30, 26, 35, 32, 29, 34,
    27, 31, 38, 25, 30, 33, 28, 36, 32, 29,
    34, 27, 31, 35, 30, 28, 33, 68, 32, 29,
    34, 27, 31, 35, 30, 71, 28, 33, 36, 32,
    29, 34, 27, 31, 75, 35, 30, 28, 33, 32,
    29, 36, 34, 27, 31, 30, 35, 28, 32, 29
])

data = {
    "OrderID": np.arange(1, 101),
    "DeliveryTime(min)": delivery_times
}

df = pd.DataFrame(data)

df.to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")

delivery = df["DeliveryTime(min)"]

mean_delivery = delivery.mean()
std_delivery = delivery.std()

print(
    f"Real delivery times -- Mean: "
    f"{mean_delivery:.1f} min "
    f"Std: {std_delivery:.1f} min"
)

simulated_delivery = np.random.normal(
    loc=mean_delivery,
    scale=std_delivery,
    size=100
)

print(
    f"Simulated 100 values from "
    f"Normal(mean={mean_delivery:.1f}, "
    f"std={std_delivery:.1f})"
)

print("\nMissing values:", delivery.isnull().sum())

q1 = delivery.quantile(0.25)
median = delivery.quantile(0.50)
q3 = delivery.quantile(0.75)

print(
    f"Percentiles -- 25th: {q1:.1f} "
    f"50th: {median:.1f} "
    f"75th: {q3:.1f}"
)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(
    f"IQR outlier bounds: "
    f"{lower_bound:.1f} - {upper_bound:.1f}"
)

outlier_mask = (
    (delivery < lower_bound) |
    (delivery > upper_bound)
)

outliers = df[outlier_mask]

print(
    f"Outliers detected: "
    f"{len(outliers)} deliveries "
    f"({outliers['OrderID'].tolist()})"
)

percentage_outliers = (
    len(outliers) / len(df)
) * 100

print(
    f"Percentage flagged as outliers: "
    f"{percentage_outliers:.1f}%"
)

plt.figure(figsize=(8, 5))

plt.hist(
    delivery,
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_delivery,
    bins=15,
    alpha=0.6,
    label="Simulated Normal"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()

plt.savefig("delivery_comparison_histogram.png")
plt.show()

plt.figure(figsize=(6, 5))

plt.boxplot(delivery)

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()

plt.savefig("delivery_boxplot.png")
plt.show()

print(
    "\nSaved delivery_comparison_histogram.png, "
    "delivery_boxplot.png"
)

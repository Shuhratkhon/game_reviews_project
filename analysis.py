import os
import ast
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("supercell_reviews_dataset.csv")

dataset["date"] = pd.to_datetime(dataset["date"])

os.makedirs("charts", exist_ok=True)


# avg ratings per year:
print("\nAVERAGE RATINGS PER YEAR\n")

avg = dataset.groupby(
    ["game", "year"]
)["rating"].mean()

for (game, year), value in avg.items():
    print(
        game,
        "-",
        year,
        "average rating:",
        round(value, 2)
    )


# rating growth/decline:
print("\nRATING GROWTH / DECLINE\n")

for game in dataset["game"].unique():

    r2024 = avg.get((game, 2024))
    r2025 = avg.get((game, 2025))
    r2026 = avg.get((game, 2026))

    if r2024 is not None and r2025 is not None:
        growth_1 = ((r2025 - r2024) / r2024) * 100
    else:
        growth_1 = None

    if r2025 is not None and r2026 is not None:
        growth_2 = ((r2026 - r2025) / r2025) * 100
    else:
        growth_2 = None

    print("\nGame:", game)

    print(
        "2024 average rating:",
        round(r2024, 2) if r2024 is not None else "no data"
    )

    print(
        "2025 average rating:",
        round(r2025, 2) if r2025 is not None else "no data"
    )

    print(
        "2026 average rating:",
        round(r2026, 2) if r2026 is not None else "no data"
    )

    if growth_1 is not None:
        print(
            "2024 → 2025 change:",
            round(growth_1, 2),
            "%"
        )

    if growth_2 is not None:
        print(
            "2025 → 2026 change:",
            round(growth_2, 2),
            "%"
        )


# creating rating growth chart:
print("\nCREATING RATING GROWTH CHART\n")

plt.figure(figsize=(12, 7))

years = [2024, 2025, 2026]

for game in dataset["game"].unique():

    values = [
        avg.get((game, year))
        for year in years
    ]

    plt.plot(
        years,
        values,
        marker="o",
        linewidth=2,
        label=game.replace("_", " ").title()
    )

    for year, value in zip(years, values):
        if value is not None:
            plt.annotate(
                f"{value:.2f}",
                (year, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center"
            )

plt.title("Supercell Games — Rating Growth / Decline")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.xticks(years)
plt.ylim(1, 5)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "charts/rating_growth.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved: charts/rating_growth.png")

# creating topic/rating pie charts:
print("\nCREATING TOPIC AND RATING PIE CHARTS\n")

topic_colors = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#92400e",
    "#ec4899",
    "#6b7280",
    "#a3a300",
    "#06b6d4"
]

rating_colors = [
    "#dc2626",
    "#f97316",
    "#f59e0b",
    "#facc15",
    "#fde047"
]

for game in dataset["game"].unique():

    game_data = dataset[dataset["game"] == game]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11)
    )

    fig.suptitle(
        game.replace("_", " ").title() + " - Topic & Rating Distribution",
        fontsize=22,
        fontweight="bold"
    )

    for column, year in enumerate([2024, 2025, 2026]):

        year_data = game_data[
            game_data["year"] == year
        ]

        topic_counts = {}

        for topics in year_data["topics"]:

            if isinstance(topics, str):
                topics = ast.literal_eval(topics)

            for topic in topics:
                topic_counts[topic] = (
                    topic_counts.get(topic, 0) + 1
                )

        topic_df = pd.Series(topic_counts)

        if not topic_df.empty:

            axes[0, column].pie(
                topic_df.values,
                labels=topic_df.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=topic_colors[:len(topic_df)]
            )

        axes[0, column].set_title(
            "Topics - " + str(year),
            fontsize=15,
            fontweight="bold"
        )

        rating_counts = (
            year_data["rating"]
            .value_counts()
            .reindex([5, 4, 3, 2, 1], fill_value=0)
        )

        axes[1, column].pie(
            rating_counts.values,
            labels=[
                "5 stars",
                "4 stars",
                "3 stars",
                "2 stars",
                "1 star"
            ],
            autopct="%1.1f%%",
            startangle=90,
            colors=[
                rating_colors[4],
                rating_colors[3],
                rating_colors[2],
                rating_colors[1],
                rating_colors[0]
            ]
        )

        axes[1, column].set_title(
            "Ratings - " + str(year),
            fontsize=15,
            fontweight="bold"
        )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    filename = (
        "charts/"
        + game
        + "_topic_rating_distribution.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", filename)

print("\nALL PIE CHARTS CREATED")


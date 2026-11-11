import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

if not os.path.exists("supercell_reviews_topics.csv"):
    raise SystemExit("Run topic_analyzer.py first, it creates supercell_reviews_topics.csv")

dataset = pd.read_csv("supercell_reviews_topics.csv")
dataset["topic_list"] = dataset["topics"].str.split("; ")

os.makedirs("charts", exist_ok=True)

years = [2024, 2025, 2026]
games = list(dataset["game"].unique())


# ---------- chart style ----------
BG = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BLUE = "#2a78d6"
RED = "#e34948"

# every game always keeps the same color
game_colors = dict(zip(games, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
year_colors = {2024: "#86b6ef", 2025: "#2a78d6", 2026: "#104281"}   # light -> dark = old -> new
star_colors = {1: "#c73a3a", 2: "#f0a09f", 3: "#cfcec8", 4: "#86b6ef", 5: "#1c5cab"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def titles(fig, title, subtitle):
    fig.text(0.015, 0.975, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.015, 0.925, subtitle, fontsize=10, color=INK_2, va="top")


def nice_name(game):
    return game.replace("_", " ").title().replace(" Of ", " of ")


# ---------- average rating per year + growth ----------
avg = dataset.groupby(["game", "year"])["rating"].mean()
size = dataset.groupby(["game", "year"]).size()

print("\nAVERAGE RATING PER YEAR")

for game in games:
    print("\n" + nice_name(game))
    previous = None

    for year in years:
        if (game, year) not in avg:
            print(f"  {year}: no data")
            previous = None
            continue

        value = avg[(game, year)]
        line = f"  {year}: {value:.2f}  ({size[(game, year)]} reviews)"

        if previous is not None:
            change = (value - previous) / previous * 100
            line += f"  change: {change:+.1f}%"

        if size[(game, year)] < 30:
            line += "  <- small sample, not reliable"

        print(line)
        previous = value


# ---------- chart 1: rating by year ----------
fig, ax = plt.subplots(figsize=(11, 6.5))
end_labels = []

for game in games:
    pts = [(y, avg[(game, y)], size[(game, y)]) for y in years if (game, y) in avg]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = game_colors[game]

    ax.plot(xs, ys, color=color, linewidth=2, label=nice_name(game), zorder=2)

    # filled dot = enough reviews, hollow dot = less than 30 reviews
    for x, y, n in pts:
        face = color if n >= 30 else BG
        ax.plot(x, y, "o", markersize=9, markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=2, zorder=3)

    end_labels.append([ys[-1], ys[-1], nice_name(game), color])

# label the last point of every line, pushed apart so they don't overlap
end_labels.sort(key=lambda item: item[0])
for i in range(1, len(end_labels)):
    if end_labels[i][1] - end_labels[i - 1][1] < 0.16:
        end_labels[i][1] = end_labels[i - 1][1] + 0.16

for value, y_pos, name, color in end_labels:
    ax.annotate(f"{name}  {value:.2f}", xy=(years[-1], value), xytext=(years[-1] + 0.08, y_pos),
                va="center", fontsize=10, color=INK, fontweight="bold")
    ax.plot([years[-1] + 0.02, years[-1] + 0.07], [value, y_pos], color=color, linewidth=1)

ax.set_xticks(years)
ax.set_xlim(years[0] - 0.15, years[-1] + 0.75)
ax.set_ylim(1, 5)
ax.set_ylabel("Average rating (stars)")
ax.grid(axis="x", visible=False)

legend_items = [Line2D([], [], color=game_colors[g], linewidth=2, label=nice_name(g)) for g in games]
legend_items += [
    Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED, label="30+ reviews"),
    Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=BG,
           markeredgecolor=MUTED, markeredgewidth=2, label="under 30 reviews"),
]
ax.legend(handles=legend_items, loc="upper left", ncol=4, fontsize=9)

titles(fig, "Average rating by year",
       "Top 100 most-liked Google Play reviews per game and year. Hollow dots have too few reviews to trust.")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("charts/rating_growth.png", dpi=300)
plt.close(fig)
print("\nSaved: charts/rating_growth.png")


# ---------- chart 2: most discussed topics ----------
topics = dataset.explode("topic_list")
topic_counts = topics["topic_list"].value_counts().sort_values()
share = topic_counts / len(dataset) * 100

fig, ax = plt.subplots(figsize=(10, 6.5))
bars = ax.barh(topic_counts.index, topic_counts.values, height=0.62, color=BLUE)
ax.bar_label(bars, labels=[f"{c}  ({s:.0f}%)" for c, s in zip(topic_counts, share)],
             padding=4, fontsize=9, color=INK_2)
ax.set_xlabel("Number of reviews")
ax.set_xlim(0, topic_counts.max() * 1.18)
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)

titles(fig, "What players talk about",
       f"Reviews mentioning each topic, all games ({len(dataset)} reviews). One review can have several topics.")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("charts/popular_topics.png", dpi=300)
plt.close(fig)
print("Saved: charts/popular_topics.png")


# ---------- chart 3: average rating per topic ----------
topic_rating = topics.groupby("topic_list")["rating"].mean().sort_values()

print("\nAVERAGE RATING BY TOPIC")
for topic, value in topic_rating.items():
    print(f"  {topic:<25} {value:.2f}")

fig, ax = plt.subplots(figsize=(10, 6.5))
colors = [RED if r < 3 else BLUE for r in topic_rating]

# bars start at 3 stars (neutral), so it's easy to see which topics are below / above
bars = ax.barh(topic_rating.index, topic_rating.values - 3, left=3, height=0.62, color=colors)
for bar, value in zip(bars, topic_rating.values):
    x = value - 0.05 if value < 3 else value + 0.05
    ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.2f}",
            ha="right" if value < 3 else "left", va="center", fontsize=9, color=INK_2)

ax.axvline(3, color="#898781", linewidth=1)
ax.set_xlim(1, 5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("Average rating (stars)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=RED, label="below 3 stars"), Patch(color=BLUE, label="3 stars or above")],
          loc="lower right")

titles(fig, "Which topics come with bad ratings",
       "Average star rating of reviews that mention each topic. Bars start at 3 stars (neutral).")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("charts/topic_avg_rating.png", dpi=300)
plt.close(fig)
print("Saved: charts/topic_avg_rating.png")


# ---------- chart 4: topics and ratings for every game ----------
for game in games:
    game_data = dataset[dataset["game"] == game]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})

    # left: % of reviews mentioning each topic, one bar per year
    order = game_data["topic_list"].explode().value_counts().index[::-1]
    bar_h = 0.26

    for i, year in enumerate(years):
        year_data = game_data[game_data["year"] == year]
        if year_data.empty:
            continue
        pct = year_data["topic_list"].explode().value_counts().reindex(order, fill_value=0) / len(year_data) * 100
        positions = [k + (i - 1) * bar_h for k in range(len(order))]
        ax1.barh(positions, pct.values, height=bar_h, color=year_colors[year],
                 edgecolor=BG, linewidth=1.5, label=f"{year} ({len(year_data)} reviews)")

    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order)
    ax1.set_xlabel("% of that year's reviews")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="y", visible=False)
    ax1.tick_params(axis="y", length=0)
    ax1.set_title("Topics", loc="left", fontsize=12, fontweight="bold")
    # legend in the same order as the bars (newest on top)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc="lower right")

    # right: star distribution per year as 100% stacked bars
    for row, year in enumerate(years[::-1]):
        year_data = game_data[game_data["year"] == year]
        if year_data.empty:
            ax2.text(50, row, "no data", ha="center", va="center", color=MUTED)
            continue

        pct = year_data["rating"].value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
        left = 0
        for star, value in pct.items():
            ax2.barh(row, value, left=left, height=0.6, color=star_colors[star], edgecolor=BG, linewidth=2)
            if value >= 7:
                text_color = "white" if star in (1, 5) else INK
                ax2.text(left + value / 2, row, f"{value:.0f}%", ha="center", va="center",
                         fontsize=9, color=text_color)
            left += value

    ax2.set_yticks(range(len(years)))
    ax2.set_yticklabels([f"{y}\n{(game_data['year'] == y).sum()} reviews" for y in years[::-1]])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("% of reviews")
    ax2.grid(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Star ratings", loc="left", fontsize=12, fontweight="bold")
    ax2.legend(handles=[Patch(color=star_colors[s], label=f"{s} star" + ("s" if s > 1 else "")) for s in range(1, 6)],
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=9)

    titles(fig, nice_name(game) + " - topics and ratings by year",
           "Left: share of reviews mentioning each topic. Right: how the star ratings are split.")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    filename = f"charts/{game}_topic_rating_distribution.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print("Saved:", filename)

print("\nAll charts are in the charts folder")

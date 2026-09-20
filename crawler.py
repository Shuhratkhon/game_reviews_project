from google_play_scraper import reviews, Sort
import pandas as pd

# main:
games = {
    "clash_of_clans": "com.supercell.clashofclans",
    "clash_royale": "com.supercell.clashroyale",
    "brawl_stars": "com.supercell.brawlstars",
    "boom_beach": "com.supercell.boombeach",
    "hay_day": "com.supercell.hayday"
}

years = [2024, 2025, 2026]

all_reviews = []

# crawling:
for game, app_id in games.items():
    print("\nCrawling reviews for:", game)
    result, _ = reviews(
        app_id,
        lang="en",
        country="us",
        sort=Sort.MOST_RELEVANT,
        count=3000
    )

    df = pd.DataFrame(result)
    df["year"] = df["at"].dt.year

    for year in years:
        year_reviews = df[df["year"] == year]
        top_reviews = year_reviews.sort_values(
            "thumbsUpCount", ascending=False
        ).head(100)

        for _, r in top_reviews.iterrows():
            all_reviews.append({
                "game": game,
                "year": year,
                "review_text": r["content"],
                "rating": r["score"],
                "date": r["at"],
                "likes": r["thumbsUpCount"]
            })

# saving:
dataset = pd.DataFrame(all_reviews)
dataset.to_csv("supercell_reviews_dataset.csv", index=False)

print("\nDataset created successfully")
print("Total reviews collected:", len(dataset))


# analysis:
print("\nAVERAGE RATINGS PER YEAR\n")

avg = dataset.groupby(["game", "year"])["rating"].mean()

for (game, year), value in avg.items():
    print(game, "-", year, "average rating:", round(value, 2))


print("\nRATING GROWTH / DECLINE\n")

for game in games.keys():
    r2024 = avg.get((game, 2024))
    r2025 = avg.get((game, 2025))
    r2026 = avg.get((game, 2026))

    if r2024 and r2025:
        growth_1 = ((r2025 - r2024) / r2024) * 100
    else:
        growth_1 = None

    if r2025 and r2026:
        growth_2 = ((r2026 - r2025) / r2025) * 100
    else:
        growth_2 = None

    print("\nGame:", game)
    print("2024 average rating:", round(r2024, 2) if r2024 else "no data")
    print("2025 average rating:", round(r2025, 2) if r2025 else "no data")
    print("2026 average rating:", round(r2026, 2) if r2026 else "no data")

    if growth_1:
        print("2024 → 2025 change:", round(growth_1, 2), "%")
    if growth_2:
        print("2025 → 2026 change:", round(growth_2, 2), "%")

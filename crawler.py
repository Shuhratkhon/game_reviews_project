from google_play_scraper import reviews, Sort
import pandas as pd

games = {
    "clash_of_clans": "com.supercell.clashofclans",
    "clash_royale": "com.supercell.clashroyale",
    "brawl_stars": "com.supercell.brawlstars",
    "boom_beach": "com.supercell.boombeach",
    "hay_day": "com.supercell.hayday"
}

years = [2024, 2025, 2026]
per_year = 100       # reviews we want for each game and year
max_reviews = 6000   # stop looking after this many reviews per game
batch_size = 200


def enough_reviews(found):
    counts = pd.Series([r["at"].year for r in found]).value_counts()
    return all(counts.get(year, 0) >= per_year for year in years)


def crawl_game(app_id):
    found = []
    token = None

    # google play gives reviews in pages, so we keep asking until
    # every year has enough reviews or there is nothing left
    while len(found) < max_reviews:
        result, token = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.MOST_RELEVANT,
            count=batch_size,
            continuation_token=token
        )

        if not result:
            break

        found.extend(result)

        if enough_reviews(found) or token is None:
            break

    df = pd.DataFrame(found)
    if df.empty:
        return df

    return df.drop_duplicates(subset="reviewId")


all_reviews = []

for game, app_id in games.items():
    print("\nCrawling reviews for:", game)

    try:
        df = crawl_game(app_id)
    except Exception as e:
        print("  failed:", e)
        continue

    if df.empty:
        print("  no reviews found")
        continue

    print("  reviews checked:", len(df))
    df["year"] = df["at"].dt.year

    for year in years:
        # most liked reviews of that year
        top = df[df["year"] == year].sort_values("thumbsUpCount", ascending=False).head(per_year)

        if len(top) < per_year:
            print(f"  warning: only {len(top)} reviews for {year}")

        for _, r in top.iterrows():
            all_reviews.append({
                "game": game,
                "year": year,
                "review_text": r["content"],
                "rating": r["score"],
                "date": r["at"],
                "likes": r["thumbsUpCount"]
            })

dataset = pd.DataFrame(all_reviews)
dataset.to_csv("supercell_reviews_dataset.csv", index=False)

print("\nDataset saved: supercell_reviews_dataset.csv")
print("Total reviews:", len(dataset))
print(dataset.groupby(["game", "year"]).size().unstack(fill_value=0))

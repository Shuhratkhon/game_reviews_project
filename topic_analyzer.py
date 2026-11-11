import re
import pandas as pd

# strong words = clearly about the topic
# weak words = only used when no strong word was found in the review
topic_vocabulary = {
    "GAMEPLAY": {
        "strong": [
            "gameplay", "control", "controls", "combat", "battle", "fight",
            "attack", "defense", "defence", "strategy", "mechanic", "difficulty",
            "pvp", "pve", "movement", "aim", "skill", "ability", "abilities",
            "weapon", "damage", "mission", "quest", "boss", "progression",
            "tutorial", "troop", "card", "deck", "brawler", "builder",
            "upgrade", "farm", "crop", "base"
        ],
        "weak": [
            "fun", "boring", "easy", "hard", "challenging", "challenge",
            "addictive", "grind", "grinding"
        ]
    },

    "MATCHMAKING & BALANCE": {
        "strong": [
            "matchmaking", "balance", "balanced", "unbalanced", "imbalance",
            "overpowered", "op", "nerf", "buff", "ranked", "rank", "ranking",
            "league", "trophy", "trophies", "ladder", "skill gap", "level gap",
            "power gap", "power level"
        ],
        "weak": [
            "match", "fair", "unfair", "competitive", "opponent", "enemy",
            "enemies", "win", "lose", "losing", "loss"
        ]
    },

    "CONTENT & UPDATES": {
        "strong": [
            "update", "content", "feature", "character", "hero", "map",
            "event", "season", "mode", "patch", "version", "release",
            "rework", "expansion"
        ],
        "weak": [
            "new", "added", "change", "recent", "latest"
        ]
    },

    "MONETIZATION": {
        "strong": [
            "$", "price", "pricing", "cost", "expensive", "purchase", "spend",
            "spending", "pay", "paying", "payment", "bundle", "offer",
            "discount", "gem", "currency", "microtransaction", "monetization",
            "battle pass", "pass royale", "brawl pass", "gold pass",
            "pay to win", "pay2win", "p2w", "real money", "free to play", "f2p"
        ],
        "weak": [
            "money", "buy", "premium", "shop", "store", "value", "cheap",
            "deal", "reward", "free"
        ]
    },

    "TECHNICAL": {
        "strong": [
            "bug", "glitch", "crash", "freeze", "frozen", "lag", "laggy",
            "error", "connection", "disconnect", "disconnected", "loading",
            "performance", "fps", "frame rate", "server", "black screen",
            "not working", "battery drain", "won't load", "doesn't load"
        ],
        "weak": [
            "slow", "problem", "broken", "device", "phone", "internet",
            "restart", "stuck", "delay", "battery", "storage"
        ]
    },

    "VISUAL & AUDIO": {
        "strong": [
            "graphics", "graphic", "visual", "animation", "design", "art",
            "artwork", "skin", "cosmetic", "outfit", "sound", "audio",
            "music", "song", "voice", "soundtrack", "resolution"
        ],
        "weak": [
            "look", "beautiful", "ugly", "color", "colour", "style", "cute"
        ]
    },

    "ADVERTISING": {
        "strong": [
            "ad", "advertisement", "advertising", "advert", "commercial",
            "popup", "pop up", "banner"
        ],
        "weak": [
            "watch", "video", "skip"
        ]
    },

    "SOCIAL": {
        "strong": [
            "clan", "guild", "teammate", "teamwork", "community", "chat",
            "message", "social", "co-op", "coop", "multiplayer", "toxic",
            "toxicity", "bullying", "clan war", "friendly battle"
        ],
        "weak": [
            "friend", "team", "together", "group", "people", "online", "member"
        ]
    },

    "ACCOUNT & SUPPORT": {
        "strong": [
            "account", "login", "log in", "sign in", "password", "username",
            "supercell id", "support", "customer service", "ticket", "refund",
            "ban", "banned", "suspended", "suspension", "verification",
            "hacked", "hacker"
        ],
        "weak": [
            "help", "service", "locked", "recover", "lost", "access",
            "contact", "respond", "response", "email"
        ]
    }
}


def make_pattern(words):
    # one regex for a whole word list, also matches plurals (bug -> bugs, crash -> crashes)
    parts = []
    for word in words:
        if word == "$":
            parts.append(r"\$")
        else:
            parts.append(r"(?<!\w)" + re.escape(word) + r"(?:s|es)?(?!\w)")
    return re.compile("|".join(parts))


patterns = {}
for topic, vocab in topic_vocabulary.items():
    patterns[topic] = {
        "strong": make_pattern(vocab["strong"]),
        "weak": make_pattern(vocab["weak"])
    }


def find_topics(review):
    if not isinstance(review, str):
        return ["OTHERS"]

    text = review.lower()

    strong = [t for t, p in patterns.items() if p["strong"].search(text)]
    if strong:
        return strong

    weak = [t for t, p in patterns.items() if p["weak"].search(text)]
    if weak:
        return weak

    return ["OTHERS"]


dataset = pd.read_csv("supercell_reviews_dataset.csv")

dataset["topics"] = dataset["review_text"].apply(lambda r: "; ".join(find_topics(r)))

# saved to a new file so the original crawled data stays untouched
dataset.to_csv("supercell_reviews_topics.csv", index=False)
print("Saved: supercell_reviews_topics.csv")

print("\nTOPIC ANALYSIS")

for game in dataset["game"].unique():
    game_data = dataset[dataset["game"] == game]
    counts = game_data["topics"].str.split("; ").explode().value_counts()

    print("\n" + "-" * 50)
    print(game.upper(), "-", len(game_data), "reviews")
    print("-" * 50)

    for topic, count in counts.items():
        print(f"{topic:<25} {count:>4}  ({count / len(game_data) * 100:.1f}% of reviews)")

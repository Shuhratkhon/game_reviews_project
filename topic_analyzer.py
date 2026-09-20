import re
import pandas as pd

# topic_vocabulary:
topic_vocabulary = {
    "GAMEPLAY": {
        "strong": [
            "gameplay", "control", "combat", "battle", "fight",
            "attack", "defense", "strategy", "mechanic", "difficulty",
            "pvp", "pve", "movement", "aim", "shoot", "skill",
            "ability", "weapon", "damage", "health", "survival",
            "mission", "quest", "objective", "turn", "round",
            "level", "boss", "progression", "tutorial", "controls",
            "game mode", "play style", "playstyle"
        ],
        "weak": [
            "play", "player", "game", "fun", "boring",
            "easy", "hard", "challenging", "challenge", "experience"
        ]
    },

    "MATCHMAKING & BALANCE": {
        "strong": [
            "matchmaking", "balance", "balanced", "unbalanced",
            "overpowered", "overpower", "op", "nerf", "buff",
            "ranked", "ranking", "rank", "league", "trophy",
            "ladder", "skill gap", "team balance",
            "matchmaking system", "rank system", "match quality",
            "power level", "power gap", "level gap", "trophy road",
            "ranked mode", "competitive mode", "matchmaking algorithm"
        ],
        "weak": [
            "match", "fair", "unfair", "competitive",
            "competition", "strong", "weak", "win", "lose",
            "loss", "opponent", "enemy"
        ]
    },

    "CONTENT & UPDATES": {
        "strong": [
            "update", "content", "feature", "character", "hero",
            "weapon", "map", "event", "season", "mode", "patch",
            "version", "release", "rework", "addition", "expansion",
            "new content", "new character", "new weapon", "new map",
            "new mode", "new event", "game mode", "season event",
            "limited event", "season update", "major update",
            "minor update", "content update", "feature update",
            "new feature", "new season", "new level", "new mission"
        ],
        "weak": [
            "new", "add", "change", "recent", "latest", "coming", "future"
        ]
    },

    "MONETIZATION": {
        "strong": [
            "$", "price", "pricing", "cost", "expensive",
            "purchase", "buy", "spend", "pay", "payment",
            "bundle", "offer", "discount", "sale", "gem",
            "cash", "currency", "microtransaction", "monetization",
            "in app purchase", "in-app purchase", "battle pass",
            "premium pass", "pay to win", "p2w", "real money",
            "paid content", "paid item", "paid feature",
            "store purchase", "purchase price", "item price",
            "shop price", "premium currency"
        ],
        "weak": [
            "money", "premium", "shop", "store", "value",
            "reward", "free", "cheap", "deal", "item"
        ]
    },

    "TECHNICAL": {
        "strong": [
            "bug", "issue", "crash", "freeze", "lag", "error",
            "glitch", "connection", "disconnect", "loading",
            "performance", "fps", "frame rate", "server",
            "black screen", "not working", "crash report",
            "connection issue", "server issue", "network issue",
            "loading screen", "loading time", "performance issue",
            "frame drop", "connection problem", "server problem",
            "technical issue", "technical problem", "game crash",
            "game freeze", "app crash", "app freeze",
            "device compatibility", "battery drain", "memory usage"
        ],
        "weak": [
            "slow", "problem", "broken", "device", "phone",
            "internet", "restart", "stuck", "delay", "work",
            "working", "screen", "battery", "storage"
        ]
    },

    "VISUAL & AUDIO": {
        "strong": [
            "graphics", "visual", "animation", "design", "art",
            "artwork", "appearance", "skin", "cosmetic", "outfit",
            "sound", "audio", "music", "song", "voice",
            "sound effect", "visual effect", "special effect",
            "character design", "map design", "graphic quality",
            "visual quality", "sound quality", "audio quality",
            "animation quality", "art style", "visual style",
            "character skin", "weapon skin", "cosmetic item",
            "background music", "voice acting", "soundtrack",
            "resolution"
        ],
        "weak": [
            "look", "beautiful", "ugly", "color", "style",
            "quality", "effect", "picture", "image"
        ]
    },

    "ADVERTISING": {
        "strong": [
            "ad", "advertisement", "advertising", "commercial",
            "popup", "banner", "video ad", "rewarded ad",
            "forced ad", "ad break", "watch ad", "skip ad",
            "ad free", "ad-free", "remove ads", "too many ads",
            "ads everywhere", "advert", "commercial break",
            "advertising system", "ad system", "ad frequency",
            "ad placement", "full screen ad", "fullscreen ad",
            "interstitial ad", "reward ad", "optional ad",
            "forced advertising", "video advertising",
            "ad popup", "ad banner"
        ],
        "weak": [
            "watch", "video", "reward", "free", "break",
            "skip", "shown", "appear", "screen"
        ]
    },

    "SOCIAL": {
        "strong": [
            "clan", "guild", "teammate", "teamwork", "community",
            "chat", "messaging", "message", "social", "co-op",
            "coop", "multiplayer", "toxicity", "toxic", "bullying",
            "report player", "team play", "clan war", "friend list",
            "voice chat", "group chat", "online multiplayer",
            "clan member", "team member", "social system",
            "clan system", "guild system", "team system",
            "community event", "player interaction",
            "social feature", "chat system"
        ],
        "weak": [
            "friend", "team", "player", "together", "group",
            "people", "online", "member"
        ]
    },

    "ACCOUNT & SUPPORT": {
        "strong": [
            "account", "login", "log in", "sign in", "password",
            "username", "profile", "support", "customer support",
            "customer service", "help desk", "ticket", "refund",
            "ban", "banned", "suspended", "suspension",
            "verification", "account recovery", "account access",
            "account problem", "login problem", "login issue",
            "support ticket", "support team", "support response",
            "lost account", "locked account", "banned account",
            "email verification", "identity verification"
        ],
        "weak": [
            "help", "service", "problem", "locked", "recover",
            "lost", "access", "contact", "respond", "response", "email"
        ]
    }
}

# analyzing:
def normalize_text(text):
    text = text.lower()

    replacements = {
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "0": "o"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def contains_term(text, term):
    if term == "$":
        return "$" in text

    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text) is not None


def find_topics(review):
    review = normalize_text(review)

    strong_topics = set()
    weak_topics = set()

    for topic, vocabulary in topic_vocabulary.items():

        for word in vocabulary["strong"]:
            if contains_term(review, word):
                strong_topics.add(topic)
                break

        for word in vocabulary["weak"]:
            if contains_term(review, word):
                weak_topics.add(topic)
                break

    if strong_topics:
        topics = strong_topics
    elif weak_topics:
        topics = weak_topics
    else:
        topics = {"OTHERS"}

    return sorted(topics)


dataset = pd.read_csv(
    "supercell_reviews_dataset.csv"
)

dataset["topics"] = dataset["review_text"].apply(find_topics)

dataset.to_csv(
    "supercell_reviews_dataset.csv",
    index=False
)

print("\nTOPIC ANALYSIS:\n")

for game in dataset["game"].unique():

    game_data = dataset[dataset["game"] == game]

    print("\n" + "-" * 50)
    print(game.upper())
    print("-" * 50)

    print("Total reviews analyzed:", len(game_data))

    topic_counts = {}

    for topics in game_data["topics"]:
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("\nTOPIC COUNTS:")

    for topic, count in sorted(
        topic_counts.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(topic, "-", count)
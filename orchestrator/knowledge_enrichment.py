#!/usr/bin/env python3
"""
AURA Knowledge Enrichment Daemon
Runs in background. Even when user doesn't talk to AURA:
1. Crawls internet for user's interests (AI, crypto, tech, hospitality)
2. Monitors X/Twitter, Reddit, HackerNews for relevant topics
3. Summarizes and stores in local vector DB (MemoryPilot)
4. Generates self-training data from new knowledge
5. AURA becomes more knowledgeable every day, without user interaction

This is the "intellectual growth" component of self-improvement.
The nightly_improve.py handles "behavioral improvement" (learning from mistakes).
This script handles "knowledge improvement" (learning new information).
"""

import os
import json
import time
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta

MEMORY_URL = "http://localhost:23100"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "aura"
AURA_HOME = Path.home() / "Cursor/App/soflution-llm"
LOG_FILE = AURA_HOME / "logs/enrichment.log"

# User's interests (auto-discovered from conversations + manually set)
INTERESTS = [
    "LLM local open source 2026",
    "Apple Silicon MLX inference optimization",
    "Substrate Polkadot blockchain development",
    "Proof of Useful Work crypto",
    "hospitality SaaS technology",
    "SvelteKit Rust WASM development",
    "Gemma 4 SuperGemma fine-tuning",
    "federated learning decentralized AI",
    "macOS developer tools productivity",
    "Saint-Tropez Calvi luxury hospitality",
]

# Sources to crawl
SOURCES = {
    "hackernews": "https://hacker-news.firebaseio.com/v0",
    "reddit_ai": "https://www.reddit.com/r/LocalLLaMA/hot.json?limit=10",
    "reddit_ml": "https://www.reddit.com/r/MachineLearning/hot.json?limit=10",
    "reddit_crypto": "https://www.reddit.com/r/substrate/hot.json?limit=10",
}

(AURA_HOME / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ENRICH] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
log = logging.getLogger("enrichment")

def fetch_url(url, timeout=10):
    """Fetch URL content safely."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "AURA-Enrichment/1.0 (local knowledge bot)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.warning(f"  Fetch failed {url}: {e}")
        return None

def save_to_memory(content, tags, kind="knowledge"):
    """Save knowledge to MemoryPilot."""
    try:
        data = json.dumps({
            "content": content[:2000],
            "kind": kind,
            "tags": tags,
        }).encode()
        req = urllib.request.Request(
            f"{MEMORY_URL}/add",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except:
        return False

def summarize_with_aura(text, context=""):
    """Use AURA itself to summarize/analyze content."""
    try:
        prompt = f"""Summarize this content concisely (3-5 sentences max). 
Extract key facts, numbers, names, and actionable insights.
Context: {context}

Content: {text[:3000]}"""
        data = json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": "5m",
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("message", {}).get("content", "")
    except:
        return ""

# ── Source 1: HackerNews top stories ──
def crawl_hackernews():
    """Fetch top HN stories and summarize relevant ones."""
    log.info("Crawling HackerNews...")
    top = fetch_url(f"{SOURCES['hackernews']}/topstories.json")
    if not top:
        return 0
    saved = 0
    for story_id in top[:30]:  # Top 30 stories
        story = fetch_url(f"{SOURCES['hackernews']}/item/{story_id}.json")
        if not story or "title" not in story:
            continue
        title = story.get("title", "")
        url = story.get("url", "")
        # Check if relevant to user interests
        title_lower = title.lower()
        relevant = any(
            any(word in title_lower for word in interest.lower().split())
            for interest in INTERESTS
        )
        if relevant:
            summary = f"[HackerNews] {title}\nURL: {url}\nScore: {story.get('score', 0)} | Comments: {story.get('descendants', 0)}"
            # Optionally summarize the actual article
            enriched = summarize_with_aura(summary, "HackerNews trending article")
            if enriched:
                summary += f"\nAURA Summary: {enriched}"
            if save_to_memory(summary, ["hackernews", "tech", "enrichment"]):
                saved += 1
                log.info(f"  Saved: {title[:60]}")
        time.sleep(0.5)  # Rate limiting
    return saved

# ── Source 2: Reddit subreddits ──
def crawl_reddit():
    """Fetch hot posts from relevant subreddits."""
    log.info("Crawling Reddit...")
    saved = 0
    for name, url in SOURCES.items():
        if not name.startswith("reddit_"):
            continue
        data = fetch_url(url)
        if not data or "data" not in data:
            continue
        posts = data.get("data", {}).get("children", [])
        for post in posts[:10]:
            p = post.get("data", {})
            title = p.get("title", "")
            selftext = p.get("selftext", "")[:500]
            score = p.get("score", 0)
            permalink = p.get("permalink", "")
            if score < 10:
                continue
            summary = f"[Reddit/{name}] {title}\nScore: {score}\nhttps://reddit.com{permalink}"
            if selftext:
                summary += f"\n{selftext[:300]}"
            enriched = summarize_with_aura(summary, f"Reddit {name} post")
            if enriched:
                summary += f"\nAURA Summary: {enriched}"
            if save_to_memory(summary, ["reddit", name, "enrichment"]):
                saved += 1
                log.info(f"  Saved: {title[:60]}")
            time.sleep(0.5)
    return saved

# ── Source 3: DuckDuckGo search for interests ──
def crawl_web_interests():
    """Search the web for each interest topic."""
    log.info("Crawling web for user interests...")
    saved = 0
    for interest in INTERESTS:
        try:
            query = urllib.parse.quote(interest)
            url = f"https://html.duckduckgo.com/html/?q={query}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "AURA-Enrichment/1.0"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            # Extract snippets
            import re
            results = re.findall(
                r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL
            )
            for snippet in results[:3]:
                clean = re.sub(r'<[^>]+>', '', snippet).strip()
                if len(clean) > 50:
                    summary = f"[Web/{interest}] {clean}"
                    if save_to_memory(summary, ["web", "enrichment", interest.split()[0].lower()]):
                        saved += 1
            time.sleep(2)  # Respect rate limits
        except Exception as e:
            log.warning(f"  Web search failed for '{interest}': {e}")
    return saved

# ── Source 4: Self-generated training from new knowledge ──
def generate_self_training():
    """Use new knowledge to create Q&A pairs for self-improvement."""
    log.info("Generating self-training from new knowledge...")
    try:
        # Fetch recent enrichment memories
        data = json.dumps({
            "query": "enrichment", "limit": 20, "kind": "knowledge"
        }).encode()
        req = urllib.request.Request(
            f"{MEMORY_URL}/search", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            memories = json.loads(resp.read()).get("results", [])
    except:
        return 0
    if not memories:
        return 0
    # Ask AURA to generate Q&A pairs from the knowledge
    knowledge_text = "\n".join([m.get("content", "")[:200] for m in memories[:10]])
    prompt = f"""Based on this recent knowledge, generate exactly 5 question-answer pairs 
that would help someone stay informed. Format as JSON array:
[{{"question": "...", "answer": "..."}}]

Knowledge:
{knowledge_text}

Return ONLY the JSON array, nothing else."""

    try:
        data = json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            content = result.get("message", {}).get("content", "")
        # Parse and save as SFT training data
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group())
                dataset_dir = AURA_HOME / "training/datasets"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                sft_path = dataset_dir / f"enrichment_{datetime.now().strftime('%Y%m%d')}.jsonl"
                with open(sft_path, "a") as f:
                    for qa in qa_pairs:
                        entry = {"messages": [
                            {"role": "user", "content": qa.get("question", "")},
                            {"role": "assistant", "content": qa.get("answer", "")},
                        ]}
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log.info(f"  Generated {len(qa_pairs)} self-training pairs")
                return len(qa_pairs)
        except:
            pass
    except Exception as e:
        log.warning(f"  Self-training generation failed: {e}")
    return 0

# ── Main enrichment pipeline ──
def run_enrichment():
    log.info("=" * 60)
    log.info("AURA Knowledge Enrichment")
    log.info(f"Started at {datetime.now().isoformat()}")
    log.info("=" * 60)

    total_saved = 0

    # Crawl sources
    total_saved += crawl_hackernews()
    total_saved += crawl_reddit()
    total_saved += crawl_web_interests()

    # Generate self-training from new knowledge
    qa_count = generate_self_training()

    log.info("=" * 60)
    log.info(f"Enrichment complete: {total_saved} knowledge items saved")
    log.info(f"Self-training pairs generated: {qa_count}")
    log.info("AURA is now more knowledgeable.")
    log.info("=" * 60)


if __name__ == "__main__":
    run_enrichment()

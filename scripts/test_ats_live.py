"""Quick live smoke test — one real slug per ATS adapter."""
import asyncio
import textwrap
from dataclasses import dataclass

import httpx

from ingestion.ats import Posting, greenhouse, lever, ashby

# One known slug per provider
TESTS = [
    ("greenhouse", "anthropic",  greenhouse.fetch_postings),
    ("lever",      "mistral",    lever.fetch_postings),
    ("ashby",      "linear",     ashby.fetch_postings),
]


@dataclass
class Result:
    ats: str
    slug: str
    count: int
    sample: Posting | None
    error: str | None


async def probe(ats: str, slug: str, fetch_fn) -> Result:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            postings = await fetch_fn(slug, client)
        return Result(ats, slug, len(postings), postings[0] if postings else None, None)
    except Exception as exc:
        return Result(ats, slug, 0, None, str(exc))


async def main() -> None:
    results = await asyncio.gather(*[probe(ats, slug, fn) for ats, slug, fn in TESTS])

    for r in results:
        print(f"\n{'='*60}")
        print(f"  {r.ats.upper()}  slug={r.slug}")
        print(f"{'='*60}")
        if r.error:
            print(f"  ERROR: {r.error}")
            continue
        print(f"  postings fetched : {r.count}")
        if r.sample:
            p = r.sample
            print(f"  sample title     : {p.title}")
            print(f"  department       : {p.department}")
            print(f"  team             : {p.team}")
            print(f"  location         : {p.location}")
            print(f"  remote           : {p.remote}")
            print(f"  employment_type  : {p.employment_type}")
            print(f"  seniority        : {p.seniority}")
            print(f"  comp             : {p.compensation_min}–{p.compensation_max} "
                  f"{p.compensation_currency} / {p.compensation_interval}")
            print(f"  updated_at       : {p.updated_at}")
            print(f"  url              : {p.url}")
            if p.description_text:
                snippet = textwrap.shorten(p.description_text, width=120, placeholder="…")
                print(f"  description      : {snippet}")


if __name__ == "__main__":
    asyncio.run(main())

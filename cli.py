import argparse

import httpx


def main():
    parser = argparse.ArgumentParser(description="Video embedding CLI")
    parser.add_argument("--host", default="http://localhost:8000")
    sub = parser.add_subparsers(dest="command")

    embed_p = sub.add_parser("embed", help="Embed a video file")
    embed_p.add_argument("video_path")
    embed_p.add_argument("--chunk-duration", type=float, default=10.0)
    embed_p.add_argument("--overlap", type=float, default=2.0)
    embed_p.add_argument("--index-dir", default="index")

    search_p = sub.add_parser("search", help="Search embedded video")
    search_p.add_argument("query")
    search_p.add_argument("--top-k", type=int, default=5)
    search_p.add_argument("--index-dir", default="index")

    args = parser.parse_args()

    if args.command == "embed":
        resp = httpx.post(f"{args.host}/embed", json={
            "video_path": args.video_path,
            "chunk_duration": args.chunk_duration,
            "overlap": args.overlap,
            "index_dir": args.index_dir,
        }, timeout=600)
        print(resp.json())

    elif args.command == "search":
        resp = httpx.get(f"{args.host}/search", params={
            "q": args.query,
            "top_k": args.top_k,
            "index_dir": args.index_dir,
        }, timeout=30)
        for r in resp.json()["results"]:
            print(f"[{r['start_sec']:.1f}s – {r['end_sec']:.1f}s] {r['transcript'][:100]}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

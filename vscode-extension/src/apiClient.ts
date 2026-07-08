export interface EmbedResult {
  status: string;
  segments: number;
}

export interface SegmentMeta {
  segment_id: number;
  start_sec: number;
  end_sec: number;
  frame_path?: string;
  transcript: string;
  video_path?: string;
}

export interface SearchResult {
  results: SegmentMeta[];
}

export interface EmbedOptions {
  chunkDuration: number;
  overlap: number;
  indexDir: string;
}

export interface SearchOptions {
  topK: number;
  indexDir: string;
}

type FetchFn = (url: string, init?: RequestInit) => Promise<Response>;

export class ApiClient {
  constructor(
    private baseUrl: string,
    private fetchFn: FetchFn = (url, init) => fetch(url, init)
  ) {}

  async embedVideo(videoPath: string, opts: EmbedOptions): Promise<EmbedResult> {
    const resp = await this.fetchFn(`${this.baseUrl}/embed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_path: videoPath,
        chunk_duration: opts.chunkDuration,
        overlap: opts.overlap,
        index_dir: opts.indexDir,
      }),
    });
    if (!resp.ok) throw new Error(`Embed failed: ${resp.status}`);
    return resp.json() as Promise<EmbedResult>;
  }

  async search(query: string, opts: SearchOptions): Promise<SearchResult> {
    const params = new URLSearchParams({
      q: query,
      top_k: String(opts.topK),
      index_dir: opts.indexDir,
    });
    const resp = await this.fetchFn(`${this.baseUrl}/search?${params}`);
    if (!resp.ok) throw new Error(`Search failed: ${resp.status}`);
    return resp.json() as Promise<SearchResult>;
  }

  async getSegment(segmentId: number, indexDir: string): Promise<SegmentMeta> {
    const params = new URLSearchParams({ index_dir: indexDir });
    const resp = await this.fetchFn(`${this.baseUrl}/segment/${segmentId}?${params}`);
    if (!resp.ok) throw new Error(`Segment ${segmentId} not found`);
    return resp.json() as Promise<SegmentMeta>;
  }

  async ping(): Promise<boolean> {
    try {
      const resp = await this.fetchFn(`${this.baseUrl}/docs`);
      return resp.ok;
    } catch {
      return false;
    }
  }
}

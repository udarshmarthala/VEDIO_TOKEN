import * as assert from "assert";
import { ApiClient } from "../../apiClient";

const mockFetch = async (url: string): Promise<Response> => {
  if (url.includes("/embed")) {
    return { ok: true, json: async () => ({ status: "ok", segments: 3 }) } as unknown as Response;
  }
  if (url.includes("/search")) {
    return {
      ok: true,
      json: async () => ({
        results: [{ segment_id: 1, start_sec: 0, end_sec: 10, transcript: "hello" }],
      }),
    } as unknown as Response;
  }
  if (url.includes("/segment/1")) {
    return {
      ok: true,
      json: async () => ({ segment_id: 1, start_sec: 0, end_sec: 10, transcript: "hello" }),
    } as unknown as Response;
  }
  return { ok: false, status: 404, json: async () => ({}) } as unknown as Response;
};

suite("ApiClient", () => {
  const client = new ApiClient("http://localhost:8000", mockFetch);

  test("embedVideo returns segment count", async () => {
    const result = await client.embedVideo("video.mp4", {
      chunkDuration: 10,
      overlap: 2,
      indexDir: "index",
    });
    assert.strictEqual(result.segments, 3);
    assert.strictEqual(result.status, "ok");
  });

  test("search returns results array", async () => {
    const result = await client.search("hello world", { topK: 5, indexDir: "index" });
    assert.strictEqual(result.results.length, 1);
    assert.strictEqual(result.results[0].segment_id, 1);
  });

  test("getSegment returns metadata", async () => {
    const seg = await client.getSegment(1, "index");
    assert.strictEqual(seg.segment_id, 1);
    assert.strictEqual(seg.transcript, "hello");
  });
});

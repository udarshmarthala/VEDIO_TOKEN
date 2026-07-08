import * as vscode from "vscode";
import { SegmentMeta } from "../apiClient";

export class ResultsPanel {
  static currentPanel: ResultsPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;

  static show(
    _extensionUri: vscode.Uri,
    query: string,
    results: SegmentMeta[]
  ): void {
    if (ResultsPanel.currentPanel) {
      ResultsPanel.currentPanel._panel.reveal(vscode.ViewColumn.One);
      ResultsPanel.currentPanel._update(query, results);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "vedioTokenResults",
      "Video Search Results",
      vscode.ViewColumn.One,
      { enableScripts: false }
    );
    ResultsPanel.currentPanel = new ResultsPanel(panel, query, results);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    query: string,
    results: SegmentMeta[]
  ) {
    this._panel = panel;
    this._update(query, results);
    this._panel.onDidDispose(() => {
      ResultsPanel.currentPanel = undefined;
    });
  }

  private _update(query: string, results: SegmentMeta[]): void {
    this._panel.webview.html = this._buildHtml(query, results);
  }

  private _buildHtml(query: string, results: SegmentMeta[]): string {
    const rows = results
      .map(
        (r) => `
        <tr>
          <td>${r.segment_id}</td>
          <td>${r.start_sec.toFixed(1)}s – ${r.end_sec.toFixed(1)}s</td>
          <td>${this._esc(r.transcript ?? "")}</td>
          <td><code>${this._esc(r.video_path ?? "")}</code></td>
        </tr>`
      )
      .join("");

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 1rem; }
  h2 { color: var(--vscode-foreground); }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 0.5rem; border-bottom: 1px solid var(--vscode-panel-border); text-align: left; }
  th { color: var(--vscode-descriptionForeground); font-size: 0.85em; text-transform: uppercase; }
  tr:hover { background: var(--vscode-list-hoverBackground); }
  code { font-family: var(--vscode-editor-font-family); font-size: 0.85em; }
</style>
</head>
<body>
<h2>Results for &ldquo;${this._esc(query)}&rdquo;</h2>
${
  results.length === 0
    ? "<p>No results found.</p>"
    : `<table>
  <thead><tr><th>#</th><th>Time</th><th>Transcript</th><th>Video</th></tr></thead>
  <tbody>${rows}</tbody>
</table>`
}
</body>
</html>`;
  }

  private _esc(s: string): string {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
}

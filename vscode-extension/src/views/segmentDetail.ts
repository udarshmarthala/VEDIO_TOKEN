import * as vscode from "vscode";
import * as path from "path";
import { SegmentMeta } from "../apiClient";

export class SegmentDetailPanel {
  static show(extensionUri: vscode.Uri, segment: SegmentMeta): void {
    const localRoots: vscode.Uri[] = [extensionUri];
    if (segment.frame_path) {
      localRoots.push(vscode.Uri.file(path.dirname(segment.frame_path)));
    }

    const panel = vscode.window.createWebviewPanel(
      "vedioTokenSegment",
      `Segment #${segment.segment_id}`,
      vscode.ViewColumn.Beside,
      { enableScripts: false, localResourceRoots: localRoots }
    );

    let frameTag = "";
    if (segment.frame_path) {
      const webviewUri = panel.webview.asWebviewUri(
        vscode.Uri.file(segment.frame_path)
      );
      frameTag = `<img src="${webviewUri}" alt="frame" style="max-width:100%;border-radius:4px;margin-bottom:1rem;">`;
    }

    const rows = (Object.entries(segment) as [string, unknown][])
      .map(
        ([k, v]) =>
          `<tr><th>${k}</th><td>${String(v).replace(/</g, "&lt;")}</td></tr>`
      )
      .join("");

    panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body { font-family: var(--vscode-font-family); padding: 1rem; }
  h2 { color: var(--vscode-foreground); }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--vscode-descriptionForeground); padding: 0.4rem 0.6rem; width: 35%; }
  td { padding: 0.4rem 0.6rem; }
  tr { border-bottom: 1px solid var(--vscode-panel-border); }
</style>
</head>
<body>
<h2>Segment #${segment.segment_id}</h2>
${frameTag}
<table><tbody>${rows}</tbody></table>
</body>
</html>`;
  }
}

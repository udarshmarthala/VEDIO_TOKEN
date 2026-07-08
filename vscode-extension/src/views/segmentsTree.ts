import * as vscode from "vscode";
import { ApiClient, SegmentMeta } from "../apiClient";
import { getConfig } from "../config";
import { SegmentDetailPanel } from "./segmentDetail";

class SegmentItem extends vscode.TreeItem {
  constructor(public readonly segment: SegmentMeta) {
    super(
      `#${segment.segment_id} — ${segment.start_sec.toFixed(1)}s–${segment.end_sec.toFixed(1)}s`,
      vscode.TreeItemCollapsibleState.None
    );
    this.description = (segment.transcript ?? "").slice(0, 60);
    this.tooltip = segment.transcript ?? "";
    this.iconPath = new vscode.ThemeIcon("play");
    this.command = {
      command: "vedio-token.openSegment",
      title: "Open Segment",
      arguments: [segment],
    };
  }
}

export class SegmentsTreeProvider
  implements vscode.TreeDataProvider<SegmentItem>
{
  private _onDidChange = new vscode.EventEmitter<
    SegmentItem | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private segments: SegmentMeta[] = [];

  constructor(private client: ApiClient) {}

  refresh(): void {
    this._loadSegments().then(() => this._onDidChange.fire());
  }

  private async _loadSegments(): Promise<void> {
    const cfg = getConfig();
    try {
      const result = await this.client.search("", {
        topK: 50,
        indexDir: cfg.indexDir,
      });
      this.segments = result.results;
    } catch {
      this.segments = [];
    }
  }

  getTreeItem(element: SegmentItem): vscode.TreeItem {
    return element;
  }

  getChildren(): SegmentItem[] {
    return this.segments.map((s) => new SegmentItem(s));
  }
}

export function registerSegmentsTree(
  context: vscode.ExtensionContext,
  client: ApiClient
): SegmentsTreeProvider {
  const provider = new SegmentsTreeProvider(client);

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("vedioTokenSegments", provider)
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vedio-token.refreshSegments", () =>
      provider.refresh()
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "vedio-token.openSegment",
      (segment: SegmentMeta) => {
        SegmentDetailPanel.show(context.extensionUri, segment);
      }
    )
  );

  return provider;
}

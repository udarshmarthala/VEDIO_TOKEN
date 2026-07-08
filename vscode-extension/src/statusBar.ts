import * as vscode from "vscode";
import { ApiClient } from "./apiClient";

export class ApiStatusBar {
  private item: vscode.StatusBarItem;
  private timer: ReturnType<typeof setInterval> | undefined;

  constructor(private client: ApiClient) {
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = "vedio-token.startServer";
    this.item.show();
  }

  startPolling(intervalMs = 10_000): void {
    this._check();
    this.timer = setInterval(() => this._check(), intervalMs);
  }

  private async _check(): Promise<void> {
    const alive = await this.client.ping();
    if (alive) {
      this.item.text = "$(check) Vedio Token";
      this.item.tooltip = "API server running";
      this.item.backgroundColor = undefined;
    } else {
      this.item.text = "$(warning) Vedio Token";
      this.item.tooltip = "API server not reachable — click to start";
      this.item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
    }
  }

  dispose(): void {
    if (this.timer !== undefined) clearInterval(this.timer);
    this.item.dispose();
  }
}

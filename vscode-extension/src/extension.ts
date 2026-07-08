import * as vscode from "vscode";
import { ApiClient } from "./apiClient";
import { getConfig } from "./config";
import { registerEmbedVideoCommand } from "./commands/embedVideo";
import { registerSearchVideoCommand } from "./commands/searchVideo";
import { registerSegmentsTree } from "./views/segmentsTree";
import { ApiStatusBar } from "./statusBar";
import { registerServerCommands } from "./server";

export function activate(context: vscode.ExtensionContext): void {
  const cfg = getConfig();
  const client = new ApiClient(cfg.apiUrl);
  registerServerCommands(context);
  registerEmbedVideoCommand(context, client);
  registerSearchVideoCommand(context, client);
  registerSegmentsTree(context, client);
  const statusBar = new ApiStatusBar(client);
  statusBar.startPolling();
  context.subscriptions.push({ dispose: () => statusBar.dispose() });
}

export function deactivate(): void {}

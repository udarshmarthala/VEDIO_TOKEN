import * as vscode from "vscode";
import { ApiClient } from "../apiClient";
import { getConfig } from "../config";
import { ResultsPanel } from "../views/resultsPanel";

export function registerSearchVideoCommand(
  context: vscode.ExtensionContext,
  client: ApiClient
): void {
  const cmd = vscode.commands.registerCommand(
    "vedio-token.searchVideo",
    async () => {
      const cfg = getConfig();

      const query = await vscode.window.showInputBox({
        prompt: "Search your video index",
        placeHolder: "e.g. someone playing guitar",
      });

      if (!query) return;

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Searching for "${query}"…`,
          cancellable: false,
        },
        async () => {
          try {
            const result = await client.search(query, {
              topK: 5,
              indexDir: cfg.indexDir,
            });
            ResultsPanel.show(context.extensionUri, query, result.results);
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Search failed: ${msg}`);
          }
        }
      );
    }
  );

  context.subscriptions.push(cmd);
}

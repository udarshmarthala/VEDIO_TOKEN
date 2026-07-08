import * as vscode from "vscode";
import { ApiClient } from "../apiClient";
import { getConfig } from "../config";

export function registerEmbedVideoCommand(
  context: vscode.ExtensionContext,
  client: ApiClient
): void {
  const cmd = vscode.commands.registerCommand(
    "vedio-token.embedVideo",
    async (uri?: vscode.Uri) => {
      const cfg = getConfig();

      let videoPath: string | undefined;

      if (uri) {
        videoPath = uri.fsPath;
      } else {
        const picked = await vscode.window.showOpenDialog({
          canSelectMany: false,
          filters: { Videos: ["mp4", "mkv", "mov", "avi", "webm"] },
          title: "Select video to embed",
        });
        videoPath = picked?.[0]?.fsPath;
      }

      if (!videoPath) return;

      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Embedding ${videoPath}…`,
          cancellable: false,
        },
        async () => {
          try {
            const result = await client.embedVideo(videoPath!, {
              chunkDuration: cfg.chunkDuration,
              overlap: cfg.overlap,
              indexDir: cfg.indexDir,
            });
            vscode.window.showInformationMessage(
              `Embedded ${result.segments} segments from ${videoPath}`
            );
            vscode.commands.executeCommand("vedio-token.refreshSegments");
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Embed failed: ${msg}`);
          }
        }
      );
    }
  );

  context.subscriptions.push(cmd);
}

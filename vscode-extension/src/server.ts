import * as vscode from "vscode";
import * as cp from "child_process";
import * as path from "path";
import { getConfig } from "./config";

let serverProcess: cp.ChildProcess | undefined;
let outputChannel: vscode.OutputChannel | undefined;

export function registerServerCommands(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("Vedio Token Server");
  context.subscriptions.push(outputChannel);

  context.subscriptions.push(
    vscode.commands.registerCommand("vedio-token.startServer", async () => {
      if (serverProcess && !serverProcess.killed) {
        vscode.window.showInformationMessage("API server already running.");
        return;
      }

      const cfg = getConfig();
      const workspaceRoot =
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

      outputChannel!.show(true);
      outputChannel!.appendLine(
        `Starting: ${cfg.pythonPath} -m uvicorn vedio_token.api:app --reload`
      );

      serverProcess = cp.spawn(
        cfg.pythonPath,
        ["-m", "uvicorn", "vedio_token.api:app", "--reload", "--host", "0.0.0.0"],
        {
          cwd: workspaceRoot,
          env: {
            ...process.env,
            PYTHONPATH: path.join(workspaceRoot, "src"),
          },
        }
      );

      serverProcess.stdout?.on("data", (d: Buffer) =>
        outputChannel!.append(d.toString())
      );
      serverProcess.stderr?.on("data", (d: Buffer) =>
        outputChannel!.append(d.toString())
      );
      serverProcess.on("exit", (code) => {
        outputChannel!.appendLine(`Server exited (code ${code})`);
        serverProcess = undefined;
      });

      vscode.window.showInformationMessage("Vedio Token API server started.");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vedio-token.stopServer", () => {
      if (!serverProcess || serverProcess.killed) {
        vscode.window.showInformationMessage("No API server running.");
        return;
      }
      serverProcess.kill();
      serverProcess = undefined;
      vscode.window.showInformationMessage("Vedio Token API server stopped.");
    })
  );

  context.subscriptions.push({
    dispose: () => {
      if (serverProcess && !serverProcess.killed) serverProcess.kill();
    },
  });
}

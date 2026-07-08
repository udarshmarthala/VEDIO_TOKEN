import * as vscode from "vscode";

export interface ExtensionConfig {
  apiUrl: string;
  indexDir: string;
  chunkDuration: number;
  overlap: number;
  pythonPath: string;
}

export function getConfig(): ExtensionConfig {
  const cfg = vscode.workspace.getConfiguration("vedio-token");
  return {
    apiUrl: cfg.get<string>("apiUrl", "http://localhost:8000"),
    indexDir: cfg.get<string>("indexDir", "index"),
    chunkDuration: cfg.get<number>("chunkDuration", 10.0),
    overlap: cfg.get<number>("overlap", 2.0),
    pythonPath: cfg.get<string>("pythonPath", "python"),
  };
}

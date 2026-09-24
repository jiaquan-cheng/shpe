const { LanguageClient } = require('vscode-languageclient/node');
const vscode = require('vscode');
const path = require('path');
const fs = require('fs');

let client;

function getPythonPath(workspaceRoot) {
    const userPath = vscode.workspace.getConfiguration('shpe').get('pythonPath');
    if (userPath && userPath !== 'python' && userPath !== 'python3') {
        return userPath;
    }
    if (workspaceRoot) {
        const isWindows = process.platform === 'win32';
        const venvPython = isWindows
            ? path.join(workspaceRoot, '.venv', 'Scripts', 'python.exe')
            : path.join(workspaceRoot, '.venv', 'bin', 'python');

        if (fs.existsSync(venvPython)) {
            return venvPython;
        }
    }
    return process.platform === 'win32' ? 'python' : 'python3';
}

function activate(context) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const workspaceRoot = workspaceFolders ? workspaceFolders[0].uri.fsPath : undefined;
    
    const pythonPath = getPythonPath(workspaceRoot);

    const serverOptions = {
        command: pythonPath, 
        args: ['-m', 'shpe.server'],
        options: {
            cwd: workspaceRoot,
            env: Object.assign({}, process.env, {
                PYTHONPATH: workspaceRoot
            })
        }
    };

    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'python' }],
    };

    client = new LanguageClient('shpe', 'shpe Language Server', serverOptions, clientOptions);
    client.start();
}

function deactivate() {
    if (!client) { return undefined; }
    return client.stop();
}

module.exports = { activate, deactivate };
const { LanguageClient } = require('vscode-languageclient/node');
const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const { execFileSync } = require('child_process');
const semver = require('semver');
const extensionVersion = require('./package.json').version;

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

function getInstalledPackageVersion(pythonPath) {
    try {
        return execFileSync(
            pythonPath,
            ['-c', "import importlib.metadata; import shpe; print(importlib.metadata.version('shpe'))"],
            { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }
        ).trim();
    } catch (error) {
        return null;
    }
}

function isPackageVersionOlder(installedVersion, editorVersion) {
    const toComparableVersion = version => {
        const release = semver.coerce(version);
        if (!release) return null;

        const publicVersion = version.split('+', 1)[0];
        const prerelease = publicVersion.match(/(alpha|a|beta|b|preview|pre|rc|c|dev)[-_.]?(\d*)$/i);
        if (!prerelease) return release.version;

        const label = prerelease[1].toLowerCase();
        const normalizedLabel = label === 'a' || label === 'alpha'
            ? 'a'
            : label === 'b' || label === 'beta'
                ? 'b'
                : label === 'dev'
                    ? '0.dev'
                    : 'rc';
        return `${release.version}-${normalizedLabel}.${prerelease[2] || '0'}`;
    };

    const installed = toComparableVersion(installedVersion);
    const editor = toComparableVersion(editorVersion);
    return installed !== null && editor !== null && semver.lt(installed, editor);
}

function activate(context) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const workspaceRoot = workspaceFolders ? workspaceFolders[0].uri.fsPath : undefined;
    
    const pythonPath = getPythonPath(workspaceRoot);

    const installedVersion = getInstalledPackageVersion(pythonPath);

    if (!installedVersion) {
        vscode.window.showErrorMessage(
            `The 'shpe' package is not installed in the selected Python environment (${pythonPath}). ` +
            "Please install it (e.g. 'pip install shpe').",
            "Open Settings"
        ).then(selection => {
            if (selection === "Open Settings") {
                vscode.commands.executeCommand('workbench.action.openSettings', 'shpe.pythonPath');
            }
        });
        return; 
    }

    if (isPackageVersionOlder(installedVersion, extensionVersion)) {
        vscode.window.showWarningMessage(
            `The installed 'shpe' package version (${installedVersion}) is older than this VS Code extension version (${extensionVersion}). ` +
            `Update it with '${pythonPath} -m pip install --upgrade shpe'.`
        );
    }

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

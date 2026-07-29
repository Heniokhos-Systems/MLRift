import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    const serverModule = context.asAbsolutePath(path.join('out', 'server.js'));

    const serverOptions: ServerOptions = {
        run: { module: serverModule, transport: TransportKind.ipc },
        debug: { module: serverModule, transport: TransportKind.ipc }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'mlrift' }],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.mlr')
        },
        // Forward the configured compiler path so `mlrift.compilerPath` works.
        initializationOptions: {
            compilerPath: workspace.getConfiguration('mlrift').get('compilerPath', 'mlrc')
        }
    };

    client = new LanguageClient('mlrift', 'MLRift Language Server', serverOptions, clientOptions);
    client.start();
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) return undefined;
    return client.stop();
}
